/* Motion-Triggered Keyword Detection for XIAO ESP32S3 Sense
 * 
 * Combines camera-based motion detection with Edge Impulse audio keyword inference.
 * When motion is detected, the microphone begins listening for keywords.
 * After a period of no motion, inference stops to save power.
 *
 * Based on Edge Impulse Arduino examples (MIT License)
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/* ========================== INCLUDES ========================== */
#include <XIAOESP32S3_Telecomm_Project_inferencing.h>
#include <I2S.h>
#include "esp_camera.h"

/* ========================== CAMERA PIN DEFINITIONS ========================== */
#define CAMERA_PIN_PWDN  -1
#define CAMERA_PIN_RESET -1
#define CAMERA_PIN_XCLK  10
#define CAMERA_PIN_SIOD  40
#define CAMERA_PIN_SIOC  39
#define CAMERA_PIN_D7    48
#define CAMERA_PIN_D6    11
#define CAMERA_PIN_D5    12
#define CAMERA_PIN_D4    14
#define CAMERA_PIN_D3    16
#define CAMERA_PIN_D2    18
#define CAMERA_PIN_D1    17
#define CAMERA_PIN_D0    15
#define CAMERA_PIN_VSYNC 38
#define CAMERA_PIN_HREF  47
#define CAMERA_PIN_PCLK  13

/* ========================== CONFIGURATION ========================== */
// Audio
#define SAMPLE_RATE     16000U
#define SAMPLE_BITS     16

// LED
#define LED_BUILT_IN    21

// Motion detection tuning
#define BLOCK_SIZE          8
#define PIXEL_THRESHOLD     30    // per-pixel brightness change to count as "different"
#define MOTION_THRESHOLD    15.0  // percentage of blocks that must change to trigger motion

// How long (ms) to keep listening after the last motion event
#define MOTION_COOLDOWN_MS  10000

// How often (ms) to check for motion
#define MOTION_CHECK_INTERVAL_MS 500

/* ========================== GLOBALS ========================== */

// --- Audio inference ---
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false;
static volatile bool record_status = false;
static TaskHandle_t captureTaskHandle = NULL;

// --- Motion detection ---
static uint8_t *prev_frame = NULL;
static size_t prev_frame_len = 0;
static bool motion_active = false;           // true while we're in "listening" mode
static unsigned long last_motion_time = 0;   // timestamp of last motion event
static unsigned long last_motion_check = 0;  // timestamp of last motion check

// --- State machine ---
enum SystemState {
    STATE_WATCHING,    // camera checking for motion, mic off
    STATE_LISTENING    // motion detected, mic on, running inference
};
static SystemState currentState = STATE_WATCHING;

/* ========================== CAMERA FUNCTIONS ========================== */

bool init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = CAMERA_PIN_D0;
    config.pin_d1       = CAMERA_PIN_D1;
    config.pin_d2       = CAMERA_PIN_D2;
    config.pin_d3       = CAMERA_PIN_D3;
    config.pin_d4       = CAMERA_PIN_D4;
    config.pin_d5       = CAMERA_PIN_D5;
    config.pin_d6       = CAMERA_PIN_D6;
    config.pin_d7       = CAMERA_PIN_D7;
    config.pin_xclk     = CAMERA_PIN_XCLK;
    config.pin_pclk     = CAMERA_PIN_PCLK;
    config.pin_vsync    = CAMERA_PIN_VSYNC;
    config.pin_href     = CAMERA_PIN_HREF;
    config.pin_sccb_sda = CAMERA_PIN_SIOD;
    config.pin_sccb_scl = CAMERA_PIN_SIOC;
    config.pin_pwdn     = CAMERA_PIN_PWDN;
    config.pin_reset    = CAMERA_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size   = FRAMESIZE_QQVGA;  // 160x120
    config.fb_count     = 2;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    Serial.println("Camera initialized.");
    return true;
}

bool check_motion() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }

    bool motion_detected = false;

    if (prev_frame != NULL && prev_frame_len == fb->len) {
        int changedBlocks = 0;
        int totalBlocks = (fb->width / BLOCK_SIZE) * (fb->height / BLOCK_SIZE);

        for (size_t y = 0; y < fb->height; y += BLOCK_SIZE) {
            for (size_t x = 0; x < fb->width; x += BLOCK_SIZE) {
                int blockDiff = 0;
                int blockPixels = 0;

                for (size_t by = 0; by < BLOCK_SIZE && (y + by) < fb->height; by++) {
                    for (size_t bx = 0; bx < BLOCK_SIZE && (x + bx) < fb->width; bx++) {
                        size_t idx = (y + by) * fb->width + (x + bx);
                        int diff = abs((int)fb->buf[idx] - (int)prev_frame[idx]);
                        if (diff > PIXEL_THRESHOLD) blockDiff++;
                        blockPixels++;
                    }
                }

                if (blockDiff > blockPixels / 2) changedBlocks++;
            }
        }

        float percentChanged = (float)changedBlocks / totalBlocks * 100.0;
        if (percentChanged > MOTION_THRESHOLD) {
            Serial.printf("Motion detected! (%.1f%% blocks changed)\n", percentChanged);
            motion_detected = true;
        }
    }

    // Store current frame as previous
    if (prev_frame == NULL) {
        prev_frame = (uint8_t *)malloc(fb->len);
    }
    if (prev_frame) {
        memcpy(prev_frame, fb->buf, fb->len);
        prev_frame_len = fb->len;
    }

    esp_camera_fb_return(fb);
    return motion_detected;
}

/* ========================== AUDIO INFERENCE FUNCTIONS ========================== */

static void audio_inference_callback(uint32_t n_bytes) {
    for (int i = 0; i < n_bytes >> 1; i++) {
        inference.buffer[inference.buf_count++] = sampleBuffer[i];

        if (inference.buf_count >= inference.n_samples) {
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {
    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    size_t bytes_read = i2s_bytes_to_read;

    while (record_status) {
        esp_i2s::i2s_read(esp_i2s::I2S_NUM_0, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

        if (bytes_read <= 0) {
            ei_printf("Error in I2S read : %d", bytes_read);
        } else {
            if (bytes_read < i2s_bytes_to_read) {
                ei_printf("Partial I2S read");
            }

            // Scale the data (otherwise the sound is too quiet)
            for (int x = 0; x < i2s_bytes_to_read / 2; x++) {
                sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 8;
            }

            if (record_status) {
                audio_inference_callback(i2s_bytes_to_read);
            } else {
                break;
            }
        }
    }
    vTaskDelete(NULL);
}

static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (inference.buffer == NULL) {
        return false;
    }

    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    ei_sleep(100);

    record_status = true;

    xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, &captureTaskHandle);

    return true;
}

static void microphone_inference_stop() {
    record_status = false;
    // Give the task time to exit cleanly
    delay(200);
    captureTaskHandle = NULL;

    if (inference.buffer != NULL) {
        free(inference.buffer);
        inference.buffer = NULL;
    }
    inference.buf_ready = 0;
    inference.buf_count = 0;
}

static bool microphone_inference_record(void) {
    // Non-blocking check with timeout so we can also monitor motion
    unsigned long start = millis();
    while (inference.buf_ready == 0) {
        delay(10);
        // Timeout after 2 seconds to allow motion check cycle
        if (millis() - start > 2000) {
            return false;
        }
    }
    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}

/* ========================== STATE TRANSITIONS ========================== */

void enter_listening_state() {
    Serial.println(">>> Entering LISTENING state — starting keyword detection...");
    currentState = STATE_LISTENING;
    last_motion_time = millis();

    if (!microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
        ei_printf("ERR: Could not allocate audio buffer (size %d)\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        currentState = STATE_WATCHING;  // Fall back
        return;
    }

    ei_printf("Microphone recording started.\n");
}

void enter_watching_state() {
    Serial.println(">>> Entering WATCHING state — stopping keyword detection...");
    microphone_inference_stop();
    currentState = STATE_WATCHING;
    digitalWrite(LED_BUILT_IN, HIGH);  // LED off
}

/* ========================== SETUP ========================== */

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("=== Motion-Triggered Keyword Detection ===");

    // LED setup
    pinMode(LED_BUILT_IN, OUTPUT);
    digitalWrite(LED_BUILT_IN, HIGH);  // Off

    // Initialize camera
    if (!init_camera()) {
        Serial.println("FATAL: Camera init failed. Halting.");
        while (1);
    }

    // Initialize I2S microphone
    I2S.setAllPins(-1, 42, 41, -1, -1);
    if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
        Serial.println("FATAL: I2S init failed. Halting.");
        while (1);
    }

    // Print Edge Impulse model info
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    Serial.println("\nSystem ready. Watching for motion...\n");
}

/* ========================== MAIN LOOP ========================== */

void loop() {
    unsigned long now = millis();

    switch (currentState) {

    // ---- STATE: WATCHING (camera active, mic off) ----
    case STATE_WATCHING:
        if (now - last_motion_check >= MOTION_CHECK_INTERVAL_MS) {
            last_motion_check = now;

            if (check_motion()) {
                enter_listening_state();
            }
        }
        break;

    // ---- STATE: LISTENING (mic active, running inference) ----
    case STATE_LISTENING:
        // Periodically check for continued motion to reset cooldown
        if (now - last_motion_check >= MOTION_CHECK_INTERVAL_MS) {
            last_motion_check = now;

            if (check_motion()) {
                last_motion_time = now;  // Reset cooldown timer
            }
        }

        // Check if cooldown expired — no motion for a while
        if (now - last_motion_time > MOTION_COOLDOWN_MS) {
            enter_watching_state();
            break;
        }

        // Run one inference cycle
        {
            bool m = microphone_inference_record();
            if (!m) {
                // Timed out waiting for audio — that's OK, just loop back
                break;
            }

            signal_t signal;
            signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
            signal.get_data = &microphone_audio_signal_get_data;
            ei_impulse_result_t result = { 0 };

            EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
            if (r != EI_IMPULSE_OK) {
                ei_printf("ERR: Failed to run classifier (%d)\n", r);
                break;
            }

            // Find the best prediction
            int pred_index = 0;
            float pred_value = 0;

            ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);

            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                ei_printf("    %s: ", result.classification[ix].label);
                ei_printf_float(result.classification[ix].value);
                ei_printf("\n");

                if (result.classification[ix].value > pred_value) {
                    pred_index = ix;
                    pred_value = result.classification[ix].value;
                }
            }

            // Act on keyword detection (index 1 = your target keyword)
            if (pred_index == 1) {
                Serial.println("*** KEYWORD DETECTED! ***");
                digitalWrite(LED_BUILT_IN, LOW);  // LED on
                enter_watching_state();
            } else {
                digitalWrite(LED_BUILT_IN, HIGH);  // LED off
            }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
            ei_printf("    anomaly score: ");
            ei_printf_float(result.anomaly);
            ei_printf("\n");
#endif
        }
        break;
    }
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
