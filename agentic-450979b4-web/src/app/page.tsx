const hardware = [
  "ESP32-CAM AI-Thinker module",
  "FTDI programmer (for first flash)",
  "5V / 2A regulated power supply",
  "1-channel relay module (5V coil, opto-isolated recommended)",
  "Momentary push button or UART console for enrollment control",
  "Electric strike / door lock or other load driven by the relay",
];

const firmwarePipeline = [
  "Capture QVGA (320×240) frames for a balance between latency and accuracy.",
  "Detect faces locally with `esp_face_detect` (single detection mode).",
  "Generate embeddings via `face_recognition_forward` from ESP-WHO.",
  "Compare embeddings stored in flash using cosine similarity ≥ 0.63.",
  "Energise the relay for 3 seconds when a match is confirmed.",
  "Signal non-matches via the status LED without toggling the relay.",
];

const enrollmentSteps = [
  "Maintenir le bouton d’enrôlement au démarrage ou envoyer `enroll` en série.",
  "Capturer 5–10 images en demandant à la personne de varier légèrement les angles.",
  "Sauvegarder chaque vecteur dans la flash avec `fr_flash_save_face_id_to_flash`.",
  "Redémarrer en mode reconnaissance pour revenir au fonctionnement nominal.",
];

const softwareBlocks = [
  {
    title: "Power-On Self Test",
    description:
      "Initialise la PSRAM, le capteur OV2640, le moteur de reconnaissance et vérifie l’état du relais.",
  },
  {
    title: "Enrollment Controller",
    description:
      "Gère l’ajout ou la suppression d’identités via un bouton ou l’interface série.",
  },
  {
    title: "Recognition Loop",
    description:
      "Traite les trames en continu, filtre les faux positifs et applique la logique de porte.",
  },
  {
    title: "Relay Driver",
    description:
      "Assure un temps d’activation fixe avec anti-rebond matériel et logiciel.",
  },
];

const arduinoSketch = `#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "fd_forward.h"
#include "fr_forward.h"
#include "fr_flash.h"
#include "dl_lib.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#define RELAY_PIN 12
#define ENROLL_PIN 13
#define STATUS_LED 33
#define MATCH_THRESHOLD 0.63f
#define RELAY_PULSE_MS 3000

static face_id_name_list st_face_list;
static bool enroll_mode = false;
static uint8_t enroll_id = 0;
static bool relay_active = false;
static uint64_t relay_deadline = 0;

void startCamera() {
  camera_config_t config = {
      .ledc_channel = LEDC_CHANNEL_0,
      .ledc_timer = LEDC_TIMER_0,
      .pin_d0 = Y2_GPIO_NUM,
      .pin_d1 = Y3_GPIO_NUM,
      .pin_d2 = Y4_GPIO_NUM,
      .pin_d3 = Y5_GPIO_NUM,
      .pin_d4 = Y6_GPIO_NUM,
      .pin_d5 = Y7_GPIO_NUM,
      .pin_d6 = Y8_GPIO_NUM,
      .pin_d7 = Y9_GPIO_NUM,
      .pin_xclk = XCLK_GPIO_NUM,
      .pin_pclk = PCLK_GPIO_NUM,
      .pin_vsync = VSYNC_GPIO_NUM,
      .pin_href = HREF_GPIO_NUM,
      .pin_sscb_sda = SIOD_GPIO_NUM,
      .pin_sscb_scl = SIOC_GPIO_NUM,
      .pin_pwdn = PWDN_GPIO_NUM,
      .pin_reset = RESET_GPIO_NUM,
      .xclk_freq_hz = 20000000,
      .frame_size = FRAMESIZE_QVGA,
      .pixel_format = PIXFORMAT_JPEG,
      .grab_mode = CAMERA_GRAB_LATEST,
      .fb_location = CAMERA_FB_IN_PSRAM,
      .jpeg_quality = 12,
      .fb_count = 1,
  };

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE("camera", "Camera init failed: 0x%x", err);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_QVGA);
  s->set_pixformat(s, PIXFORMAT_RGB565);
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);
  pinMode(ENROLL_PIN, INPUT_PULLUP);
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(STATUS_LED, LOW);

  startCamera();
  face_id_init(&st_face_list, FACE_ID_SAVE_NUMBER, false);
  read_face_id_from_flash(&st_face_list);

  ESP_LOGI("boot", "Faces en mémoire: %d", st_face_list.count);
}

void energiseRelay(uint32_t duration_ms) {
  relay_active = true;
  relay_deadline = esp_timer_get_time() / 1000ULL + duration_ms;
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(STATUS_LED, HIGH);
}

void updateRelay() {
  if (!relay_active) return;
  uint64_t now = esp_timer_get_time() / 1000ULL;
  if (now >= relay_deadline) {
    relay_active = false;
    digitalWrite(RELAY_PIN, LOW);
    digitalWrite(STATUS_LED, LOW);
  }
}

void loop() {
  updateRelay();

  if (digitalRead(ENROLL_PIN) == LOW) {
    enroll_mode = true;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGW("camera", "Frame non disponible");
    return;
  }

  dl_matrix3du_t *rgb = dl_matrix3du_alloc(1, fb->width, fb->height, 3);
  fmt2rgb888(fb->buf, fb->len, fb->format, rgb->item);

  box_array_t *boxes = face_detect(rgb, &mtmn_config);
  if (!boxes) {
    dl_matrix3du_free(rgb);
    esp_camera_fb_return(fb);
    return;
  }

  if (enroll_mode) {
    int8_t remaining = enroll_face_id_to_flash(&st_face_list, boxes, rgb, enroll_id);
    if (remaining == 0) {
      ESP_LOGI("enroll", "ID %d enregistré", enroll_id);
      enroll_mode = false;
      enroll_id++;
    } else {
      ESP_LOGI("enroll", "Captures restantes: %d", remaining);
    }
  } else {
    face_id_node *best = recognize_face(&st_face_list, boxes, rgb);
    float score = best ? best->similarity : 0.0f;
    if (best && score >= MATCH_THRESHOLD) {
      ESP_LOGI("match", "Visage #%d score %.2f", best->id, score);
      if (!relay_active) {
        energiseRelay(RELAY_PULSE_MS);
      }
    } else {
      ESP_LOGI("match", "Aucun profil reconnu (%.2f)", score);
    }
  }

  dl_matrix3du_free(rgb);
  dl_lib_free(boxes);
  esp_camera_fb_return(fb);
}
`;

const maintenanceTasks = [
  "Effectuer un auto-test hebdomadaire en lisant le score d’un visage de référence.",
  "Sauvegarder la partition flash d’enrôlement avant toute mise à jour firmware.",
  "Installer l’ensemble dans un boîtier IP54 avec aération contrôlée.",
  "Protéger la charge du relais avec diode de roue libre ou snubber RC.",
];

export default function Home() {
  return (
    <div className="bg-slate-950 text-slate-100">
      <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-16 px-6 py-16 sm:px-10 lg:px-20">
        <header className="space-y-6">
          <p className="text-sm font-semibold uppercase tracking-[0.4em] text-emerald-400">
            ESP32-CAM Offline Access Control
          </p>
          <h1 className="text-4xl font-black leading-tight text-white sm:text-5xl">
            Reconnaissance faciale autonome avec activation de relais hors
            ligne.
          </h1>
          <p className="max-w-4xl text-lg text-slate-300 sm:text-xl">
            Guide complet pour construire un système d’accès basé sur l’ESP32-CAM
            utilisant la reconnaissance faciale locale via ESP-WHO, sans dépendre
            d’une connexion réseau.
          </p>
        </header>

        <section className="grid gap-6 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur">
          <h2 className="text-2xl font-semibold text-white">Matériel</h2>
          <ul className="grid gap-3 text-slate-200 sm:grid-cols-2">
            {hardware.map((item) => (
              <li
                key={item}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm"
              >
                {item}
              </li>
            ))}
          </ul>
        </section>

        <section className="grid gap-6 rounded-3xl border border-emerald-500/30 bg-emerald-500/10 p-8">
          <h2 className="text-2xl font-semibold text-emerald-200">
            Pipeline embarqué
          </h2>
          <ol className="grid gap-4 text-sm text-emerald-50 sm:grid-cols-2">
            {firmwarePipeline.map((step, index) => (
              <li
                key={step}
                className="flex gap-4 rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-4"
              >
                <span className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-emerald-500/50 text-base font-semibold text-white">
                  {index + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </section>

        <section className="grid gap-4 rounded-3xl border border-white/10 bg-white/5 p-8 text-slate-200">
          <h2 className="text-2xl font-semibold text-white">
            Architecture logicielle
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {softwareBlocks.map((block) => (
              <article
                key={block.title}
                className="rounded-2xl border border-white/10 bg-slate-900/70 p-6 shadow-lg shadow-black/30"
              >
                <h3 className="text-lg font-semibold text-white">
                  {block.title}
                </h3>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {block.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section className="grid gap-6 rounded-3xl border border-white/10 bg-slate-900/80 p-8">
          <div>
            <h2 className="text-2xl font-semibold text-white">
              Enrôlement sécurisé
            </h2>
            <p className="mt-2 text-sm text-slate-300">
              Les identités sont stockées localement en flash et survivent aux
              redémarrages. Changez le seuil de similarité pour ajuster la
              sécurité.
            </p>
          </div>
          <ol className="grid gap-4 text-sm text-slate-200 sm:grid-cols-2">
            {enrollmentSteps.map((step, index) => (
              <li
                key={step}
                className="flex gap-4 rounded-2xl border border-white/10 bg-slate-900/70 p-4"
              >
                <span className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-white/10 text-base font-semibold text-white">
                  {index + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </section>

        <section className="grid gap-6 rounded-3xl border border-emerald-400/10 bg-black/60 p-8">
          <div className="space-y-3">
            <h2 className="text-2xl font-semibold text-emerald-200">
              Firmware Arduino / ESP-IDF
            </h2>
            <p className="text-sm text-emerald-50/80">
              Compiler avec l’ESP32 board manager (≥ 3.0.0) ou le composant
              ESP-WHO sous ESP-IDF. Le sketch reste 100 % hors ligne.
            </p>
          </div>
          <pre className="overflow-x-auto rounded-2xl border border-emerald-500/30 bg-slate-950 p-6 text-xs text-emerald-50">
            <code>{arduinoSketch}</code>
          </pre>
        </section>

        <section className="grid gap-4 rounded-3xl border border-white/15 bg-white/5 p-8 text-slate-200">
          <h2 className="text-2xl font-semibold text-white">Maintenance</h2>
          <ul className="grid gap-3 sm:grid-cols-2">
            {maintenanceTasks.map((task) => (
              <li
                key={task}
                className="rounded-xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm leading-6"
              >
                {task}
              </li>
            ))}
          </ul>
        </section>

        <footer className="flex flex-col gap-3 border-t border-white/10 pt-8 text-sm text-slate-400 sm:flex-row sm:items-center sm:justify-between">
          <p>
            Solution autonome : aucune dépendance cloud, la décision se prend au
            bord.
          </p>
          <a
            href="https://github.com/espressif/esp-who"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-emerald-300 hover:text-emerald-200"
          >
            Ressources ESP-WHO
          </a>
        </footer>
      </main>
    </div>
  );
}
