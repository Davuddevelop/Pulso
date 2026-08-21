/*
 * Pulso hardware capture firmware -- ELEGOO Mega 2560 (ATmega2560).
 *
 * Samples one analog input at exactly SAMPLE_RATE_HZ (2000, matching the
 * project's fixed internal rate -- see sources.py / CLAUDE.md) and streams
 * raw 10-bit ADC values to the host over USB serial in a small binary
 * framing protocol. All the actual signal processing (filtering, envelope,
 * beat detection) happens on the laptop in dsp.py/segment.py -- this
 * sketch's only job is to get clean, correctly-timed samples off the board.
 *
 * UNTESTED ON REAL HARDWARE. Written from the datasheet and known-good AVR
 * timer/ADC patterns, not verified against an actual hoard. Sanity-check it
 * with tools/check_serial_source.py before wiring it into app.py.
 *
 * ---- Wiring ----
 * Analog signal in -> A0.
 *
 * If the sensor is a labeled breakout with VCC/GND/AO/DO pins: VCC->5V,
 * GND->GND, AO->A0. Nothing else needed.
 *
 * If it is a bare piezo disc (two wire leads, no board attached): the ADC
 * only reads 0-5V, but a piezo swings both positive and negative around
 * zero, so it needs a DC bias to sit mid-range instead of clipping at 0V.
 * Two same-value resistors (100-220k ohm) as a divider hold A0 at ~2.5V;
 * the piezo's own internal capacitance couples its AC signal onto that
 * node without needing a separate coupling capacitor:
 *
 *   5V ---[R]--- A0 ---[R]--- GND
 *                 |
 *      piezo lead 1 (piezo lead 2 -> GND)
 *
 * A piezo struck hard can spike well past 5V. If you have small signal
 * diodes (1N4148 or similar) on hand, clamp A0 to the rails for safety:
 *   A0 -> diode anode -> 5V   (conducts if A0 tries to exceed ~5.3V)
 *   GND -> diode anode -> A0  (conducts if A0 tries to go below ~-0.3V)
 * Optional but recommended; the bias resistors alone give some protection.
 *
 * ---- Wire protocol ----
 * Two bytes per sample, little-endian, value 0-1023 (10-bit ADC), so the
 * high byte is always in [0,3]. Every SYNC_INTERVAL-th sample is preceded
 * by a 2-byte marker 0xFF 0xFE -- a value neither byte of a real sample can
 * ever take, since the high byte tops out at 3 -- so the host can always
 * find a byte-aligned, verified-correct sample even if it started reading
 * mid-stream or missed bytes. Cost: 2 extra bytes every 256 samples,
 * negligible bandwidth.
 */

const uint8_t ANALOG_PIN = A0;
const uint16_t SAMPLE_RATE_HZ = 2000;
const uint16_t SYNC_INTERVAL = 256;
const uint32_t BAUD = 115200;

volatile bool sampleDue = false;
uint16_t sampleCount = 0;

// Timer1 in CTC mode, prescaler 8: 16MHz / 8 / 1000Hz = 2000 -> fires at
// 2 * SAMPLE_RATE_HZ if OCR1A = that value... derive properly below.
void setupTimer1() {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;
  // CTC mode, prescaler 8: timer clock = 16,000,000 / 8 = 2,000,000 Hz.
  // OCR1A = timer_clock / SAMPLE_RATE_HZ - 1.
  OCR1A = (2000000UL / SAMPLE_RATE_HZ) - 1;  // = 999 at 2000 Hz
  TCCR1B |= (1 << WGM12);              // CTC
  TCCR1B |= (1 << CS11);               // prescaler 8
  TIMSK1 |= (1 << OCIE1A);             // enable compare-match interrupt
  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  sampleDue = true;
}

void setup() {
  Serial.begin(BAUD);
  analogReference(DEFAULT);  // 5V reference
  setupTimer1();
}

void loop() {
  if (!sampleDue) return;
  sampleDue = false;

  uint16_t value = analogRead(ANALOG_PIN);  // 0-1023, ~100us on a Mega

  if (sampleCount % SYNC_INTERVAL == 0) {
    Serial.write(0xFF);
    Serial.write(0xFE);
  }
  Serial.write((uint8_t)(value & 0xFF));         // low byte
  Serial.write((uint8_t)((value >> 8) & 0xFF));  // high byte, always 0-3

  sampleCount++;
}
