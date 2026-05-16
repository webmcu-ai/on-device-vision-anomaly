// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// ON-DEVICE VISION ANOMALY DETECTION — v001
//
// Based on on-device-vision-classification v44
// Anomaly strategy: CNN front-end (unchanged) + Global Average Pooling +
//   multiple normal prototypes + normalized distance scoring + temporal smoothing
//
// Workflow:
//   1. Collect "normal" images into the single "0Normal" class folder
//   2. Train → computes GAP feature mean/std per prototype cluster (k-means)
//   3. Infer → live normality score 0–100% with temporal smoothing
//
// SD card stores: images in /images/0Normal/
// SD card stores: weights (CNN) and prototypes in /header/
// Serial monitor and OLED output
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique),
//   Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
//
// For platformio you need the U8g2 library declared in the platformio.ini file and OPI PSRAM set
// lib_deps =  olikraus/U8g2 @ ^2.35.30
// build_flags =
//    -DBOARD_HAS_PSRAM
//    -DARDUINO_USB_CDC_ON_BOOT=1
// board_build.arduino.memory_type = qio_opi
// board_build.flash_mode = qio
// board_upload.flash_size = 8MB
// ======================================================


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// optional Uncomment AFTER copying myWeights.h from SD to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include <U8g2lib.h>
#include <Wire.h>

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// ANOMALY DETECTION CONFIGURATION
// ======================================================

// Single class: only "normal" images are collected
#define NUM_CLASSES 1
String myClassLabels[NUM_CLASSES] = {"0Normal"};

// menu: collect normal | train | infer
const int myTotalItems = NUM_CLASSES + 2;

// CNN hyperparameters (same as classification v44)
float LEARNING_RATE = 0.0003;
int BATCH_SIZE = 6;
int TARGET_EPOCHS = 20;
int VALIDATION_IMAGES = 3;

// Anomaly-specific hyperparameters
#define GAP_SIZE       CONV2_FILTERS   // global average pooling output = one float per filter (8)
#define NUM_PROTOTYPES 3               // k-means cluster centres representing "normal"
#define KMEANS_ITERS   10              // Lloyd iterations during prototype computation
#define TEMPORAL_ALPHA 0.7f            // smoothing: S_t = alpha*S_{t-1} + (1-alpha)*score_t
#define ANOMALY_THRESHOLD 0.40f        // below this normality score = anomaly (shown on OLED)

const int myThresholdPress   = 1100;
const int myThresholdRelease = 900;


// ======================================================
// UNIFIED TOUCH INPUT SYSTEM
// ======================================================
struct TouchState {
  bool isTouching        = false;
  int  tapCount          = 0;
  unsigned long firstTapTime    = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime   = 0;
  const unsigned long tapWindow    = 800;
  const int           longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};

TouchState myTouch;

// SYSTEM LOGIC VARIABLES
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;
int  myMenuIndex     = 1;
bool myIsSelected    = false;
bool myWeightsTrained = false;

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40
#define SIOC_GPIO_NUM   39
#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15
#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13

// ======================================================
// CONFIGURABLE INPUT RESOLUTION
// ======================================================
#define INPUT_SIZE 64

// ======================================================
// CNN ARCHITECTURE CONSTANTS (identical to classification v44)
// ======================================================
#define CONV1_KERNEL_SIZE 3
#define CONV1_FILTERS     4
#define CONV1_WEIGHTS     (CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * 3 * CONV1_FILTERS)

#define CONV2_KERNEL_SIZE 3
#define CONV2_FILTERS     8
#define CONV2_WEIGHTS     (CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * 4 * CONV2_FILTERS)

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)
#define FLATTENED_SIZE    (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)

// Anomaly head replaces the old dense output_w/output_b.
// We keep one "class" for the CNN loss so that the CNN still trains with
// a meaningful signal (autoencoder-style: label=0 always).
#define OUTPUT_WEIGHTS (FLATTENED_SIZE * NUM_CLASSES)

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================
uint8_t* myRgbBuffer = nullptr;
bool     mySDavailable = false;

// ML Buffers (PSRAM)
float* myInputBuffer  = nullptr;
float* myConv1_w      = nullptr;
float* myConv1_b      = nullptr;
float* myConv2_w      = nullptr;
float* myConv2_b      = nullptr;
float* myOutput_w     = nullptr;   // kept for optional reconstruction head
float* myOutput_b     = nullptr;

// Gradient buffers
float* myConv1_w_grad  = nullptr;
float* myConv1_b_grad  = nullptr;
float* myConv2_w_grad  = nullptr;
float* myConv2_b_grad  = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

// Adam momentum buffers
float* myConv1_w_m   = nullptr; float* myConv1_w_v   = nullptr;
float* myConv1_b_m   = nullptr; float* myConv1_b_v   = nullptr;
float* myConv2_w_m   = nullptr; float* myConv2_w_v   = nullptr;
float* myConv2_b_m   = nullptr; float* myConv2_b_v   = nullptr;
float* myOutput_w_m  = nullptr; float* myOutput_w_v  = nullptr;
float* myOutput_b_m  = nullptr; float* myOutput_b_v  = nullptr;

// Forward pass buffers
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;   // size NUM_CLASSES (=1)

// Backward pass buffers
float* myDense_grad  = nullptr;
float* myConv2_grad  = nullptr;
float* myPool1_grad  = nullptr;
float* myConv1_grad  = nullptr;

// ======================================================
// ANOMALY HEAD GLOBALS
// ======================================================
// GAP feature vector for the current frame (computed in myComputeGAP)
float myGapVector[GAP_SIZE];

// Prototypes: NUM_PROTOTYPES centres of size GAP_SIZE
float myPrototypeMean[NUM_PROTOTYPES][GAP_SIZE];
float myPrototypeStd [NUM_PROTOTYPES][GAP_SIZE];
bool  myPrototypesReady = false;

// Temporal smoothing state
float mySmoothedScore = 0.5f;   // initial neutral score

struct TrainingItem {
  String path;
  int    label;
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn=-100, float mx=100) {
  if(isnan(v)||isinf(v)) return 0;
  return constrain(v,mn,mx);
}
inline float leaky_relu(float x)        { return x>0 ? x : 0.1f*x; }
inline float leaky_relu_deriv(float x)  { return x>0 ? 1.0f : 0.1f; }

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching     = false;
  myTouch.tapCount       = 0;
  myTouch.firstTapTime   = 0;
  myTouch.lastReleaseTime= 0;
  myTouch.lastCheckTime  = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);

  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount  = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching      = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count  = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else             Serial.printf("TAP detected (%d tap%s)\n", count, count>1?"s":"");
      return result;
    }
  }
  return 0;
}

void myCheckTouchBackground() { myUpdateTouchState(); }

int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching)
    if (now - myTouch.firstTapTime > myTouch.tapWindow)
      return (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
  return 0;
}


// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;

  Serial.println("\n=== Allocating Memory ===");

  myInputBuffer  = (float*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
  myConv1_w      = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b      = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w      = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b      = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w     = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b     = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_w_grad  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b_grad  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w_grad  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b_grad  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w_grad = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b_grad = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_w_m  = (float*)ps_calloc(CONV1_WEIGHTS, sizeof(float));
  myConv1_w_v  = (float*)ps_calloc(CONV1_WEIGHTS, sizeof(float));
  myConv1_b_m  = (float*)ps_calloc(CONV1_FILTERS, sizeof(float));
  myConv1_b_v  = (float*)ps_calloc(CONV1_FILTERS, sizeof(float));
  myConv2_w_m  = (float*)ps_calloc(CONV2_WEIGHTS, sizeof(float));
  myConv2_w_v  = (float*)ps_calloc(CONV2_WEIGHTS, sizeof(float));
  myConv2_b_m  = (float*)ps_calloc(CONV2_FILTERS, sizeof(float));
  myConv2_b_v  = (float*)ps_calloc(CONV2_FILTERS, sizeof(float));
  myOutput_w_m = (float*)ps_calloc(OUTPUT_WEIGHTS, sizeof(float));
  myOutput_w_v = (float*)ps_calloc(OUTPUT_WEIGHTS, sizeof(float));
  myOutput_b_m = (float*)ps_calloc(NUM_CLASSES, sizeof(float));
  myOutput_b_v = (float*)ps_calloc(NUM_CLASSES, sizeof(float));

  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myDense_grad = (float*)ps_malloc(FLATTENED_SIZE * sizeof(float));
  myConv2_grad = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myPool1_grad = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv1_grad = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while(1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // He initialisation
  float c1std = sqrt(2.0/(9.0*3));
  for(int i=0;i<CONV1_WEIGHTS;i++) myConv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2.0f*c1std;
  for(int i=0;i<CONV1_FILTERS;i++) myConv1_b[i]=0;

  float c2std = sqrt(2.0/36.0);
  for(int i=0;i<CONV2_WEIGHTS;i++) myConv2_w[i]=((float)rand()/RAND_MAX-0.5f)*2.0f*c2std;
  for(int i=0;i<CONV2_FILTERS;i++) myConv2_b[i]=0;

  float dstd = sqrt(2.0/FLATTENED_SIZE);
  for(int i=0;i<OUTPUT_WEIGHTS;i++) myOutput_w[i]=((float)rand()/RAND_MAX-0.5f)*2.0f*dstd;
  for(int i=0;i<NUM_CLASSES;i++) myOutput_b[i]=0;
  Serial.println("He-init random weights set");
}

// ======================================================
// WEIGHT SAVE/LOAD  (CNN weights only — prototypes saved separately)
// ======================================================
void myExportHeader() {
  if (!mySDavailable) { Serial.println("No SD card - cannot export header"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// Anomaly model — bake-in weights after training");
  file.printf( "//   #define NUM_CLASSES %d\n", NUM_CLASSES);
  file.println("// Then uncomment:  #define USE_BAKED_WEIGHTS");
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for(int i=0;i<size;i++){
      file.print(data[i],6); file.print("f");
      if(i<size-1) file.print(", ");
      if((i+1)%8==0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDump("myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Header exported to /header/myWeights.h");
}

bool myLoadWeights() {
  if (!mySDavailable) { Serial.println("No SD - skipping weight load"); return false; }
  if (!SD.exists("/header/myWeights.bin")) { Serial.println("No SD weights found"); return false; }
  Serial.println("Loading CNN weights from SD...");
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS*4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS*4);
  f.read((uint8_t*)myConv2_w,  CONV2_WEIGHTS*4);
  f.read((uint8_t*)myConv2_b,  CONV2_FILTERS*4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS*4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES*4);
  f.close();
  Serial.println("CNN weights loaded");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) { Serial.println("No SD - cannot save weights"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS*4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS*4);
    f.write((uint8_t*)myConv2_w,  CONV2_WEIGHTS*4);
    f.write((uint8_t*)myConv2_b,  CONV2_FILTERS*4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS*4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES*4);
    f.close();
    Serial.println("CNN weights saved to SD");
  }
  myExportHeader();
}

// ======================================================
// PROTOTYPE SAVE/LOAD  (/header/myPrototypes.bin)
// Layout: NUM_PROTOTYPES * GAP_SIZE floats (mean) then same again (std)
// ======================================================
void mySavePrototypes() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myPrototypes.bin", FILE_WRITE);
  if (!f) return;
  f.write((uint8_t*)myPrototypeMean, NUM_PROTOTYPES*GAP_SIZE*sizeof(float));
  f.write((uint8_t*)myPrototypeStd,  NUM_PROTOTYPES*GAP_SIZE*sizeof(float));
  f.close();
  Serial.println("Prototypes saved to /header/myPrototypes.bin");
}

bool myLoadPrototypes() {
  if (!mySDavailable) return false;
  if (!SD.exists("/header/myPrototypes.bin")) return false;
  File f = SD.open("/header/myPrototypes.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myPrototypeMean, NUM_PROTOTYPES*GAP_SIZE*sizeof(float));
  f.read((uint8_t*)myPrototypeStd,  NUM_PROTOTYPES*GAP_SIZE*sizeof(float));
  f.close();
  myPrototypesReady = true;
  Serial.println("Prototypes loaded from SD");
  return true;
}

// ======================================================
// IMAGE LOADING FROM SD
// ======================================================
bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if(!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if(!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  if(!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if(!ok) return false;
  for(int y=0;y<INPUT_SIZE;y++) {
    for(int x=0;x<INPUT_SIZE;x++) {
      int sy = (int)((y+0.5)*240.0/INPUT_SIZE); if(sy>239)sy=239;
      int sx = (int)((x+0.5)*240.0/INPUT_SIZE); if(sx>239)sx=239;
      int srcIdx = (sy*240+sx)*3;
      int dstIdx = (y*INPUT_SIZE+x)*3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  return true;
}

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();

// ======================================================
// SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 Anomaly Detection Starting ===");
  Serial.printf("Free heap:  %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(240*240*3);
  if (!myRgbBuffer) Serial.println("Failed to allocate RGB buffer!");

  pinMode(A0, INPUT);
  u8g2.begin();

  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);

  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) { s->set_hmirror(s, 1); }

  esp_log_level_set("*",          ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  myAllocateMemory();

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  CONV2_WEIGHTS  * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  CONV2_FILTERS  * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in CNN weights loaded");
  myWeightsTrained = true;
#endif

  if (myLoadWeights())     Serial.println("SD CNN weights loaded");
  if (myLoadPrototypes())  Serial.println("SD prototypes loaded - ready to infer");

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready. Tap A0 to navigate, 3+ taps to select.");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION FUNCTIONS                                      ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;
  u8g2.firstPage();
  do {
    for (int ox = 0; ox < myOledWidth; ox++) {
      for (int oy = 0; oy < myOledHeight; oy++) {
        size_t pi = ((oy*myScaleY)*240+(ox*myScaleX))*3;
        uint8_t bright = (myRgbBuffer[pi]+myRgbBuffer[pi+1]+myRgbBuffer[pi+2])/3;
        if (bright > 100) u8g2.drawPixel(ox, oy);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0); u8g2.drawBox(0,0,20,15);
      u8g2.setColorIndex(1); u8g2.setCursor(3,10); u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0); u8g2.drawBox(50,0,22,8);
      u8g2.setColorIndex(1); u8g2.drawStr(52,7,"LIVE");
    }
  } while (u8g2.nextPage());
}

void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) return;
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) return;
  myRenderRgbToOLED(imageCount);
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    u8g2.firstPage();
    do { u8g2.drawStr(0,15,"No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2 taps) = Capture image");
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture, 'L'=exit");
  myResetTouchState();

  String path = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path))     SD.mkdir(path);

  int counts[NUM_CLASSES] = {};
  File root = SD.open("/images/" + myClassLabels[classIdx]);
  if(root) {
    while(File file = root.openNextFile()) {
      if(!file.isDirectory()) {
        String fn = String(file.name());
        if(fn.endsWith(".jpg")||fn.endsWith(".JPG")) counts[classIdx]++;
      }
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0, lastOLED = 0;
  bool oledNeedsUpdate = false, shouldCapture = false;

  while (true) {
    unsigned long now = millis();
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;
      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              oledNeedsUpdate = true; lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }
    if (oledNeedsUpdate) { oledNeedsUpdate = false; myRenderRgbToOLED(-1); }

    if (Serial.available()) {
      char c = Serial.read();
      if (c=='l'||c=='L') { myResetMenuState(); return; }
      else if (c=='t'||c=='T') shouldCapture = true;
    }

    int touchAction = myCheckTouchInput();
    if      (touchAction == 2) { myResetMenuState(); return; }
    else if (touchAction == 1) shouldCapture = true;

    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len); file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          myDisplayImageOnOLED(fb, counts[classIdx]);
          delay(300); lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }
    delay(5);
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: CNN FORWARD / BACKWARD PASS + OPTIMIZER                        ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// FORWARD PASS  (identical conv/pool path as classification v44)
// myConv2_output is left populated so myComputeGAP can use it.
// ======================================================
void myForwardPass(float* input, float* logits) {
  // Conv1
  for(int f=0;f<CONV1_FILTERS;f++) {
    int ob=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for(int y=0;y<CONV1_OUTPUT_SIZE;y++) {
      for(int x=0;x<CONV1_OUTPUT_SIZE;x++) {
        float sum=0;
        for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) {
          int inPos=((y+ky)*INPUT_SIZE+(x+kx))*3;
          int wPos=f*27+ky*9+kx*3;
          sum+=input[inPos]*myConv1_w[wPos]+input[inPos+1]*myConv1_w[wPos+1]+input[inPos+2]*myConv1_w[wPos+2];
        }
        myConv1_output[ob+y*CONV1_OUTPUT_SIZE+x]=leaky_relu(clip_value(sum+myConv1_b[f]));
      }
    }
  }
  // Pool1
  for(int f=0;f<CONV1_FILTERS;f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0;y<POOL1_OUTPUT_SIZE;y++) {
      for(int x=0;x<POOL1_OUTPUT_SIZE;x++) {
        int iy=y*2, ix=x*2;
        float maxVal=myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix];
        maxVal=max(maxVal,myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1]);
        maxVal=max(maxVal,myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix]);
        maxVal=max(maxVal,myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]);
        myPool1_output[ob+y*POOL1_OUTPUT_SIZE+x]=maxVal;
      }
    }
  }
  // Conv2
  for(int f=0;f<CONV2_FILTERS;f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0;y<CONV2_OUTPUT_SIZE;y++) {
      for(int x=0;x<CONV2_OUTPUT_SIZE;x++) {
        float sum=0;
        for(int c=0;c<CONV1_FILTERS;c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++)
            sum+=myPool1_output[ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx)]*myConv2_w[f*36+c*9+ky*3+kx];
        }
        myConv2_output[ob+y*CONV2_OUTPUT_SIZE+x]=leaky_relu(clip_value(sum+myConv2_b[f]));
      }
    }
  }
  // Dense (1 output — used only during CNN training as a trivial single-class head)
  for(int c=0;c<NUM_CLASSES;c++) {
    double sum=0, comp=0;
    for(int i=0;i<FLATTENED_SIZE;i++) {
      double term=myConv2_output[i]*myOutput_w[c*FLATTENED_SIZE+i];
      double yy=term-comp; double t=sum+yy; comp=(t-sum)-yy; sum=t;
    }
    myDense_output[c]=clip_value((float)sum+myOutput_b[c],-50,50);
  }
  // Sigmoid output (binary normal/not — only used for CNN training signal)
  float sig = 1.0f/(1.0f+exp(-myDense_output[0]));
  logits[0]         = myDense_output[0];
  myDense_output[0] = sig;
}

// ======================================================
// GLOBAL AVERAGE POOLING (called after myForwardPass)
// Compresses CONV2_OUTPUT_SIZE^2 spatial cells per filter
// into one mean value per filter → GAP_SIZE floats total.
// ======================================================
void myComputeGAP() {
  int cells = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
  for(int f=0; f<CONV2_FILTERS; f++) {
    int base = f * cells;
    float sum = 0;
    for(int i=0; i<cells; i++) sum += myConv2_output[base+i];
    myGapVector[f] = sum / cells;
  }
}

// ======================================================
// ANOMALY SCORE from GAP vector + stored prototypes
// Returns normality score in [0,1] (1 = very normal).
// ======================================================
float myComputeNormalityScore() {
  float bestDist = 1e30f;
  for(int p=0; p<NUM_PROTOTYPES; p++) {
    float dist = 0;
    for(int i=0; i<GAP_SIZE; i++) {
      float diff = (myGapVector[i] - myPrototypeMean[p][i]) / (myPrototypeStd[p][i] + 1e-4f);
      dist += diff * diff;
    }
    dist = sqrt(dist / GAP_SIZE);   // normalised RMS distance
    if (dist < bestDist) bestDist = dist;
  }
  return exp(-bestDist);            // 1 = perfectly normal, → 0 = very anomalous
}

// ======================================================
// BACKWARD PASS  (same as classification v44)
// ======================================================
void myBackwardDense(int label) {
  memset(myDense_grad, 0, FLATTENED_SIZE*sizeof(float));
  for(int c=0;c<NUM_CLASSES;c++) {
    float error = myDense_output[c] - (c==label ? 1.0f : 0.0f);
    for(int i=0;i<FLATTENED_SIZE;i++) {
      myOutput_w_grad[c*FLATTENED_SIZE+i] += error*myConv2_output[i];
      myDense_grad[i] += error*myOutput_w[c*FLATTENED_SIZE+i];
    }
    myOutput_b_grad[c] += error;
  }
}

void myBackwardConv2() {
  for(int i=0;i<FLATTENED_SIZE;i++)
    myConv2_grad[i]=myDense_grad[i]*leaky_relu_deriv(myConv2_output[i]);
  memset(myPool1_grad,0,POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  for(int f=0;f<CONV2_FILTERS;f++) {
    int ob=f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0;y<CONV2_OUTPUT_SIZE;y++) {
      for(int x=0;x<CONV2_OUTPUT_SIZE;x++) {
        float grad=myConv2_grad[ob+y*CONV2_OUTPUT_SIZE+x];
        myConv2_b_grad[f]+=grad;
        for(int c=0;c<CONV1_FILTERS;c++) {
          int ib=c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) {
            int pi=ib+(y+ky)*POOL1_OUTPUT_SIZE+(x+kx);
            int wi=f*36+c*9+ky*3+kx;
            myConv2_w_grad[wi]+=grad*myPool1_output[pi];
            myPool1_grad[pi]  +=grad*myConv2_w[wi];
          }
        }
      }
    }
  }
}

void myBackwardPool1() {
  memset(myConv1_grad,0,CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  for(int f=0;f<CONV1_FILTERS;f++) {
    int ib=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE, ob=f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0;y<POOL1_OUTPUT_SIZE;y++) {
      for(int x=0;x<POOL1_OUTPUT_SIZE;x++) {
        int iy=y*2, ix=x*2;
        float poolVal=myPool1_output[ob+y*POOL1_OUTPUT_SIZE+x];
        float grad   =myPool1_grad  [ob+y*POOL1_OUTPUT_SIZE+x];
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix]   ==poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix]   +=grad;
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1] ==poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix+1] +=grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix]==poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix]+=grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]==poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]+=grad;
      }
    }
  }
}

void myBackwardConv1() {
  for(int i=0;i<CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS;i++)
    myConv1_grad[i]*=leaky_relu_deriv(myConv1_output[i]);
  for(int f=0;f<CONV1_FILTERS;f++) {
    int ob=f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    for(int y=0;y<CONV1_OUTPUT_SIZE;y++) {
      for(int x=0;x<CONV1_OUTPUT_SIZE;x++) {
        float grad=myConv1_grad[ob+y*CONV1_OUTPUT_SIZE+x];
        myConv1_b_grad[f]+=grad;
        for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++) {
          int inPos=((y+ky)*INPUT_SIZE+(x+kx))*3;
          int wPos=f*27+ky*9+kx*3;
          myConv1_w_grad[wPos]  +=grad*myInputBuffer[inPos];
          myConv1_w_grad[wPos+1]+=grad*myInputBuffer[inPos+1];
          myConv1_w_grad[wPos+2]+=grad*myInputBuffer[inPos+2];
        }
      }
    }
  }
}

void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1=0.9f, b2=0.999f, eps=1e-6f;
  float lr_t=LEARNING_RATE*sqrt(1-pow(b2,step))/(1-pow(b1,step));
  for(int i=0;i<size;i++) {
    m[i]=b1*m[i]+(1-b1)*g[i];
    v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
    w[i]-=lr_t*m[i]/(sqrt(v[i])+eps);
    w[i]=clip_value(w[i],-10,10);
  }
}

void myUpdateWeights(int step) {
  myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  step);
  myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  step);
  myAdamUpdate(myConv2_w,  myConv2_w_grad,  myConv2_w_m,  myConv2_w_v,  CONV2_WEIGHTS,  step);
  myAdamUpdate(myConv2_b,  myConv2_b_grad,  myConv2_b_m,  myConv2_b_v,  CONV2_FILTERS,  step);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    step);
}

// ======================================================
// K-MEANS PROTOTYPE COMPUTATION
// Called once after CNN training using all normal images.
// Fills myPrototypeMean[NUM_PROTOTYPES][GAP_SIZE] and
//       myPrototypeStd [NUM_PROTOTYPES][GAP_SIZE].
// ======================================================
void myComputePrototypes(std::vector<TrainingItem>& normalImages) {
  int N = normalImages.size();
  if (N == 0) { Serial.println("No normal images for prototype computation"); return; }

  Serial.printf("Computing %d prototypes from %d normal images...\n", NUM_PROTOTYPES, N);

  // Collect all GAP vectors
  float* myAllGap = (float*)ps_malloc(N * GAP_SIZE * sizeof(float));
  if (!myAllGap) { Serial.println("PSRAM error: cannot allocate GAP matrix"); return; }

  int validCount = 0;
  for (int n = 0; n < N; n++) {
    if (!myLoadImageFromFile(normalImages[n].path.c_str(), myInputBuffer)) continue;
    float logits[NUM_CLASSES];
    myForwardPass(myInputBuffer, logits);
    myComputeGAP();
    for (int i = 0; i < GAP_SIZE; i++)
      myAllGap[validCount*GAP_SIZE + i] = myGapVector[i];
    validCount++;
    if (n % 10 == 0) Serial.printf("  GAP extracted: %d/%d\n", n, N);
  }

  int K = min(NUM_PROTOTYPES, validCount);

  // --- Initialise cluster centres (evenly spaced across sorted images) ---
  int myAssign[validCount];  // cluster assignment per sample
  for (int k = 0; k < K; k++) {
    int src = (k * validCount) / K;
    for (int i = 0; i < GAP_SIZE; i++)
      myPrototypeMean[k][i] = myAllGap[src*GAP_SIZE + i];
  }

  // --- Lloyd iterations ---
  for (int iter = 0; iter < KMEANS_ITERS; iter++) {
    // Assignment step
    for (int n = 0; n < validCount; n++) {
      float bestDist = 1e30f; int bestK = 0;
      for (int k = 0; k < K; k++) {
        float d = 0;
        for (int i = 0; i < GAP_SIZE; i++) {
          float diff = myAllGap[n*GAP_SIZE+i] - myPrototypeMean[k][i];
          d += diff*diff;
        }
        if (d < bestDist) { bestDist=d; bestK=k; }
      }
      myAssign[n] = bestK;
    }
    // Update step
    float newMean[NUM_PROTOTYPES][GAP_SIZE] = {};
    int   clusterCnt[NUM_PROTOTYPES] = {};
    for (int n = 0; n < validCount; n++) {
      int k = myAssign[n];
      for (int i = 0; i < GAP_SIZE; i++) newMean[k][i] += myAllGap[n*GAP_SIZE+i];
      clusterCnt[k]++;
    }
    for (int k = 0; k < K; k++)
      if (clusterCnt[k] > 0)
        for (int i = 0; i < GAP_SIZE; i++)
          myPrototypeMean[k][i] = newMean[k][i] / clusterCnt[k];
  }

  // --- Compute per-cluster std ---
  float sumSq[NUM_PROTOTYPES][GAP_SIZE] = {};
  int   clusterCnt2[NUM_PROTOTYPES] = {};
  for (int n = 0; n < validCount; n++) {
    int k = myAssign[n];
    for (int i = 0; i < GAP_SIZE; i++) {
      float diff = myAllGap[n*GAP_SIZE+i] - myPrototypeMean[k][i];
      sumSq[k][i] += diff*diff;
    }
    clusterCnt2[k]++;
  }
  for (int k = 0; k < K; k++) {
    for (int i = 0; i < GAP_SIZE; i++) {
      float variance = (clusterCnt2[k] > 1) ? sumSq[k][i]/(clusterCnt2[k]-1) : 1.0f;
      myPrototypeStd[k][i] = max(sqrt(variance), 0.01f);  // floor to avoid /0
    }
    Serial.printf("Prototype %d: %d members\n", k, clusterCnt2[k]);
  }

  free(myAllGap);
  myPrototypesReady = true;
  Serial.println("Prototype computation done");
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: TRAINING FUNCTION                                               ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot train");
    u8g2.firstPage();
    do { u8g2.drawStr(0,15,"No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Training mode (anomaly)");
  Serial.println("  Phase 1: CNN trains on normal images (label=0)");
  Serial.println("  Phase 2: Prototype computation via k-means on GAP vectors");
  Serial.println("  Serial: 'L'=exit during training");
  myResetTouchState();

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0,12,"TRAINING MODE");
    u8g2.drawStr(0,24,"Loading...");
  } while (u8g2.nextPage());

  if (myLoadWeights()) Serial.println("Continuing from saved CNN weights");
  else                 Serial.println("Starting fresh CNN training");

  while (true) {
    // ---- Load all normal images ----
    myTrainingData.clear();
    File root = SD.open("/images/" + myClassLabels[0]);
    if (root) {
      while(File file = root.openNextFile()) {
        if(!file.isDirectory()) {
          String fn = String(file.name());
          if(fn.endsWith(".jpg")||fn.endsWith(".JPG"))
            myTrainingData.push_back({file.path(), 0});  // label=0 (normal)
        }
        file.close();
      }
      root.close();
    }

    if (myTrainingData.empty()) {
      u8g2.firstPage();
      do { u8g2.drawStr(0,20,"No Images!"); } while (u8g2.nextPage());
      delay(2000); myResetMenuState(); return;
    }

    // Sort for deterministic split
    std::sort(myTrainingData.begin(), myTrainingData.end(),
      [](const TrainingItem& a, const TrainingItem& b){ return a.path < b.path; });

    // Validation split
    std::vector<TrainingItem> myValidationData;
    if (VALIDATION_IMAGES > 0) {
      int total_n = myTrainingData.size();
      int skip_n  = min(VALIDATION_IMAGES, total_n);
      std::vector<TrainingItem> trainOnly;
      int seen = 0;
      for (int i=(int)myTrainingData.size()-1; i>=0; i--) {
        if (seen < skip_n) { myValidationData.push_back(myTrainingData[i]); seen++; }
        else               { trainOnly.push_back(myTrainingData[i]); }
      }
      myTrainingData = trainOnly;
      Serial.printf("Train: %d  Val: %d\n",(int)myTrainingData.size(),(int)myValidationData.size());
    }

    int total           = myTrainingData.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches    = TARGET_EPOCHS * batchesPerEpoch;
    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);

    std::vector<int> indices;
    for(int i=0;i<total;i++) indices.push_back(i);

    float runningLoss = 0; int lossCount = 0;

    // ---- Phase 1: CNN training ----
    for(int batch=0; batch<totalBatches; batch++) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c=='x'||c=='X'||c=='l'||c=='L') {
          mySaveWeights(); myWeightsTrained=true; myResetMenuState(); return;
        }
      }
      myCheckTouchBackground();
      if (myPeekTouchAction()==2) {
        myCheckTouchInput();
        mySaveWeights(); myWeightsTrained=true; myResetMenuState(); return;
      }

      if(batch%batchesPerEpoch==0) {
        int epoch=batch/batchesPerEpoch+1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, TARGET_EPOCHS);
        for(int i=total-1;i>0;i--) {
          int j=random(i+1); int tmp=indices[i]; indices[i]=indices[j]; indices[j]=tmp;
        }
      }

      int batchStart=(batch%batchesPerEpoch)*BATCH_SIZE;
      int batchEnd=min(batchStart+BATCH_SIZE,total);
      float batchLoss=0;

      memset(myConv1_w_grad, 0,CONV1_WEIGHTS *sizeof(float));
      memset(myConv1_b_grad, 0,CONV1_FILTERS *sizeof(float));
      memset(myConv2_w_grad, 0,CONV2_WEIGHTS *sizeof(float));
      memset(myConv2_b_grad, 0,CONV2_FILTERS *sizeof(float));
      memset(myOutput_w_grad,0,OUTPUT_WEIGHTS*sizeof(float));
      memset(myOutput_b_grad,0,NUM_CLASSES   *sizeof(float));

      for(int i=batchStart;i<batchEnd;i++) {
        int idx=indices[i];
        if(!myLoadImageFromFile(myTrainingData[idx].path.c_str(),myInputBuffer)) continue;
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);
        // BCE loss vs label=0 (all images are normal)
        float loss = -log(max(myDense_output[0], 1e-7f));
        batchLoss += loss;
        myBackwardDense(0);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();
        if (i%3==0) {
          myCheckTouchBackground();
          if (myPeekTouchAction()==2) {
            myCheckTouchInput();
            mySaveWeights(); myWeightsTrained=true; myResetMenuState(); return;
          }
          if (Serial.available()) {
            char c=Serial.read();
            if(c=='x'||c=='X'||c=='l'||c=='L') {
              mySaveWeights(); myWeightsTrained=true; myResetMenuState(); return;
            }
          }
        }
      }

      myUpdateWeights(batch+1);
      float avgLoss = batchLoss/(batchEnd-batchStart);
      runningLoss+=avgLoss; lossCount++;

      if((batch+1)%5==0) {
        float displayLoss=runningLoss/lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setCursor(0,12); u8g2.print("Training CNN");
          u8g2.setCursor(0,24);
          u8g2.print("B:"); u8g2.print(batch+1);
          u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0,36);
          u8g2.print("L:"); u8g2.print(displayLoss,3);
        } while (u8g2.nextPage());
        runningLoss=0; lossCount=0;
      }
      if((batch+1)%10==0)
        Serial.printf("Batch %d/%d - Loss: %.4f\n", batch+1, totalBatches, avgLoss);
    }

    Serial.println("\n--- CNN Training Complete ---");

    // Validation
    if (!myValidationData.empty()) {
      float valLoss=0; int valCount=0;
      for (auto& vitem : myValidationData) {
        if(!myLoadImageFromFile(vitem.path.c_str(),myInputBuffer)) continue;
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer,logits);
        valLoss += -log(max(myDense_output[0],1e-7f));
        valCount++;
      }
      if(valCount>0)
        Serial.printf("Validation Loss: %.4f (%d images)\n", valLoss/valCount, valCount);
    }

    mySaveWeights();
    myWeightsTrained = true;

    // ---- Phase 2: Prototype computation ----
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.drawStr(0,12,"Proto compute");
      u8g2.drawStr(0,24,"Please wait...");
    } while (u8g2.nextPage());

    myComputePrototypes(myTrainingData);
    mySavePrototypes();

    u8g2.firstPage();
    do {
      u8g2.drawStr(0,12,"DONE!");
      u8g2.drawStr(0,24,"Tap:Again");
      u8g2.drawStr(0,36,"3+Taps:Exit");
    } while (u8g2.nextPage());

    myResetTouchState();
    Serial.println("Training + prototype done. T=again  L=exit");

    while (true) {
      if (Serial.available()) {
        char c=Serial.read();
        if(c=='x'||c=='X'||c=='l'||c=='L') { myResetMenuState(); return; }
        else if(c=='t'||c=='T') break;
      }
      int touchAction=myCheckTouchInput();
      if(touchAction==2) { myResetMenuState(); return; }
      else if(touchAction==1) break;
      delay(10);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: INFERENCE — LIVE ANOMALY SCORE                                  ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myActionInfer() {
  if (!myWeightsTrained || !myPrototypesReady) {
    Serial.println("ERROR: Train first (menu item 2).");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0,12,"Train first!");
      u8g2.drawStr(0,24,"(menu item 2)");
    } while (u8g2.nextPage());
    delay(3000); myResetMenuState(); return;
  }
  Serial.println("\n>>> Anomaly Inference mode");
  Serial.println("  Score 100% = very normal   0% = anomaly");
  Serial.println("  Serial: T or L = exit");
  myResetTouchState();
  mySmoothedScore = 0.5f;

  // Pre-compute resize lookup tables
  static int sy_lookup[INPUT_SIZE], sx_lookup[INPUT_SIZE];
  static bool lookup_initialized = false;
  if (!lookup_initialized) {
    for(int i=0;i<INPUT_SIZE;i++) {
      sy_lookup[i]=min((int)((i+0.5)*240.0/INPUT_SIZE),239);
      sx_lookup[i]=min((int)((i+0.5)*240.0/INPUT_SIZE),239);
    }
    lookup_initialized=true;
  }

  unsigned long frameTimes[10];
  int frameIndex=0;

  while (true) {
    unsigned long frameStart=millis();

    if (Serial.available()) {
      char c=Serial.read();
      if(c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; }
    }

    camera_fb_t* fb=esp_camera_fb_get();
    if (!fb) { delay(10); continue; }
    if (!myRgbBuffer) { esp_camera_fb_return(fb); delay(10); continue; }

    int pct = (int)(mySmoothedScore * 100.0f);
    bool isAnomaly = (mySmoothedScore < ANOMALY_THRESHOLD);

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      // Resize to INPUT_SIZE
      for(int y=0;y<INPUT_SIZE;y++) {
        int sy=sy_lookup[y];
        for(int x=0;x<INPUT_SIZE;x++) {
          int sx=sx_lookup[x];
          int srcIdx=(sy*240+sx)*3;
          int dstIdx=(y*INPUT_SIZE+x)*3;
          myInputBuffer[dstIdx]  =myRgbBuffer[srcIdx]  *0.003921569f;
          myInputBuffer[dstIdx+1]=myRgbBuffer[srcIdx+1]*0.003921569f;
          myInputBuffer[dstIdx+2]=myRgbBuffer[srcIdx+2]*0.003921569f;
        }
      }

      // Forward pass + GAP + anomaly score
      float logits[NUM_CLASSES];
      myForwardPass(myInputBuffer, logits);
      myComputeGAP();
      float rawScore = myComputeNormalityScore();

      // Temporal smoothing
      mySmoothedScore = TEMPORAL_ALPHA*mySmoothedScore + (1.0f-TEMPORAL_ALPHA)*rawScore;
      pct = (int)(mySmoothedScore * 100.0f);
      isAnomaly = (mySmoothedScore < ANOMALY_THRESHOLD);

      // OLED: every 10th frame draw image + anomaly overlay
      if (frameIndex == 9) {
        int oW=u8g2.getDisplayWidth(), oH=u8g2.getDisplayHeight();
        int scX=240/oW, scY=240/oH;
        u8g2.firstPage();
        do {
          // Downsampled camera image
          for(int ox=0;ox<oW;ox++) {
            for(int oy=0;oy<oH;oy++) {
              int pi=((oy*scY)*240+(ox*scX))*3;
              uint8_t bright=(myRgbBuffer[pi]+myRgbBuffer[pi+1]+myRgbBuffer[pi+2])/3;
              if(bright>100) u8g2.drawPixel(ox,oy);
            }
          }
          // Score bar at bottom
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0,oH-9,oW,9);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf,sizeof(buf),"%s %d%%", isAnomaly?"ANOM":"NORM", pct);
          u8g2.drawStr(1,oH-1,buf);
        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    frameTimes[frameIndex]=millis()-frameStart;
    float fps2=1000.0/frameTimes[frameIndex];
    Serial.printf("Frame %d: %lu ms (%.1f FPS)  Score: %d%%  %s\n",
      frameIndex+1, frameTimes[frameIndex], fps2, pct,
      isAnomaly?"<<ANOMALY>>":"");
    frameIndex++;

    if (frameIndex>=10) {
      int touchVal=myReadTouch();
      if(touchVal>myThresholdPress) {
        delay(200); myResetMenuState(); return;
      }
      frameIndex=0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 5: MENU SYSTEM                                                     ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myResetMenuState() {
  myIsSelected=false;
  myResetTouchState();
  myLastActivityTime=millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for(int i=1;i<=myTotalItems;i++) {
    String label=(i<=NUM_CLASSES)?myClassLabels[i-1]:
                 (i==NUM_CLASSES+1)?"Train":"Infer";
    if(i==myMenuIndex) Serial.print(" > ");
    else               Serial.print("   ");
    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Commands: t=next (tap)  l=select (longpress)");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0,8,"TAP:Next HOLD:Ok");
    int myStartItem=(myMenuIndex<=NUM_CLASSES)?1:myMenuIndex-2;
    for(int i=0;i<3;i++) {
      int cur=myStartItem+i;
      if(cur>myTotalItems) break;
      String label=(cur<=NUM_CLASSES)?myClassLabels[cur-1]:
                   (cur==NUM_CLASSES+1)?"Train":"Infer";
      int y=18+i*9;
      if(cur==myMenuIndex) u8g2.drawStr(0,y,("> "+label).c_str());
      else                 u8g2.drawStr(0,y,("  "+label).c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if(idx<=NUM_CLASSES)        myActionCollect(idx-1);
  else if(idx==NUM_CLASSES+1) myActionTrain();
  else                        myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis=millis();

  if (!myIsSelected && Serial.available()) {
    char c=Serial.read();
    if(c>='1'&&c<='9') {
      int newIndex=c-'0';
      if(newIndex<=myTotalItems) {
        myMenuIndex=newIndex; myIsSelected=true;
        myLastActivityTime=myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    } else if(c=='t'||c=='T') {
      if(myCurrentMillis-myLastTapTime>myTapCooldown) {
        myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1;
        myDrawMenu(); myLastTapTime=myCurrentMillis; myLastActivityTime=myCurrentMillis;
      }
    } else if(c=='l'||c=='L') {
      myIsSelected=true; myLastActivityTime=myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  if (!myIsSelected) {
    int touchAction=myCheckTouchInput();
    if(touchAction==1) {
      if(myCurrentMillis-myLastTapTime>myTapCooldown) {
        myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1;
        myDrawMenu(); myLastTapTime=myCurrentMillis; myLastActivityTime=myCurrentMillis;
      }
    } else if(touchAction==2) {
      myIsSelected=true; myLastActivityTime=myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}
