#include <Arduino.h>
#include "MovingAvgF.h"
#include "MovingMedianF.h"
#include <math.h>

// ====================================
//#define USE_ARDUINO
#define USE_TEENSY
// ====================================

struct MotorPins {
#if defined(USE_ARDUINO)
  const int pwmPin   = 9;
  const int in1Pin   = 7;
  const int in2Pin   = 8;
  const int encoderA = 2;
  const int encoderB = 3;
#elif defined(USE_TEENSY)
  const int pwmPin   = 23; // EN pin
  const int in1Pin   = 22; // DIR pin, cambiar valor cambia la dirección de giro
  const int in2Pin   = 21; // !SLEEP pin, active high
  const int encoderA = 15;
  const int encoderB = 14;
#else
  #error "Tarjeta no soportada"
#endif
};

struct ControlParams {
  float x[5] = {0, 0, 0, 0, 0};
  float y[5] = {0, 0, 0, 0, 0};
  float error = 0;
  float integral_sum = 0;     // para PID posicional
  float dFiltered = 0;        // para derivada filtrada
  float previousError = 0;
  float previousOutput = 0;
  float controladorOUT = 0;
  const int PPR = 240; // Pulsos por revolución encoder
  int PWMOUT = 0;
  int ZM = 10; // ZM es la zona muerta del motor, se identifica con System Ident, con onda triangular
  float A_cont = 0;
  float B_cont = 0;
  float C_cont = 0;
  float D_cont = 0;
  float E_cont = 0;
  float F_cont = 0;
  float G_cont = 0;
  float H_cont = 0;
  float Kp = 0;
  float Ki = 0;
  float Kd = 0;
  float alfa = 0.01;
  float time_constant = 0.01; // Time constant for PID algorithm derivative action filtering
  int PIDtype = 0;
  float reset_time = 0.10f;
  float d_b0, d_b1, d_a1, d_y;   // D coefficients and state
  float d_out_prev = 0.0f;  // estado para derivada Tustin en PID incremental
};

struct SystemVariables {
  bool inicio = false;
  bool tiporef = 1; //Referencia automática o manual. 1 es automática, 0 es manual
  int modooperacion = 0;    // 0: sin acción, 1: identificación, 2: control de velocidad, 3: control de posición
  int tab_activo = 0;       // 1: PID, 2: controlador digital
  int tiposenal = 0;        // 0: PRBS, 1: Cuadrada, 2: Senoidal, etc.
  int referencia = 0;
  int medicion = 0;
  int amplitudAuto = 0;
  int offset = 0;
  int referenciaManual = 0;
  int pwmMinimoLinealidad = 40;
  int pwmMaximoLinealidad = 240;
  int deg = 0;
};

struct TimingData {
  unsigned long tiempo = 0;
  unsigned long tiempoinicio = 0;
  unsigned long tiempoanterior = micros();
  unsigned long tiemporestart = 0;
  float tiempociclo = 0;
  float tiempoloop = 0;
  unsigned long int tiemporeferencia = 2000;
  int delayintencional = 10;
};

struct EncoderData {
  volatile int32_t encoderTicks = 0;
  int32_t encoderTicksActual = 0;
  int32_t encoderTicksAnterior = 0;
  int32_t lastEncoded = 0;
};

struct SerialData {
  String DatoSerial;
};

struct MeasurementsData {
  float RPM = 0;
  float RPMPromedio = 0;
  float current = 0;
  float currentMedian = 0;
  float currentAvgAux = 0;
  float currentAvg = 0;
};

MotorPins pins;
ControlParams ctrl;
SystemVariables sys;
TimingData timing;
EncoderData enc;
SerialData serialData;
MeasurementsData meas;

int avgwindowsize = 1;
int currentwindowsize = 4; //la medición de corriente es particularmente ruidosa
MovingAvgF RPMAvg(avgwindowsize);
MovingMedianF FilteredCurrentMedian(avgwindowsize); //Moving median
MovingAvgF FilteredCurrent(currentwindowsize);  // 1st order avg filter
MovingAvgF FilteredCurrent2(currentwindowsize); // 2nd order avg filter

void divideString(String DatoSerial, char delimitador, String salida[], int limite) {
  int posicion = 0;
  int indice = 0;
  while ((posicion = DatoSerial.indexOf(delimitador)) != -1 && indice < limite - 1) {
    salida[indice++] = DatoSerial.substring(0, posicion);
    DatoSerial = DatoSerial.substring(posicion + 1);
  }
  salida[indice] = DatoSerial;
}

void inicializaVariables() {
  sys.referencia = 0;
  sys.medicion = 0;
  timing.tiempo = 0;
  timing.tiempoanterior = 0;
  timing.tiempociclo = 0;
  enc.encoderTicks = 0;
  ctrl.controladorOUT = 0;
  ctrl.error = 0;
  ctrl.previousError = 0;
  ctrl.previousOutput = 0;
  enc.lastEncoded = 0;
  sys.deg = 0;
  ctrl.d_y = 0.0f;          // estado D del posicional
  ctrl.d_out_prev = 0.0f;   // estado D del incremental
  ctrl.integral_sum = 0.0f; // por si acaso

  RPMAvg.reset();
  FilteredCurrent.reset();
  FilteredCurrent2.reset();
}

void updateEncoder() {
  // Una explicación del funcionamiento de esto está disponible en:
  // https://en.wikipedia.org/wiki/Incremental_encoder
  // Sección state transitions
  int MSB = digitalRead(pins.encoderA);
  int LSB = digitalRead(pins.encoderB);
  int encoded = (MSB << 1) | LSB;
  int sum = (enc.lastEncoded << 2) | encoded;
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011)
    enc.encoderTicks++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000)
    enc.encoderTicks--;
  enc.lastEncoded = encoded;
}

float PRBS() {
    static bool is_zero = false; // Track current phase
    static unsigned long last_switch = 0;

    unsigned long now = timing.tiempo;

    // Duration for each phase
    unsigned long phase_duration = timing.tiemporeferencia / 2;

    if ((now - last_switch) >= phase_duration) {
        last_switch = now;
        is_zero = !is_zero; // Toggle phase

        if (!is_zero) {
            // On new value phase, pick random value
            sys.referencia = random(sys.pwmMinimoLinealidad, sys.pwmMaximoLinealidad);
        }
    }

    return is_zero ? 0 : sys.referencia;
}

float Cuadrada() {
  if (fmod(timing.tiempo, timing.tiemporeferencia) < timing.tiemporeferencia / 2)
    return sys.amplitudAuto + sys.offset;
  else
    return sys.offset;
}

float Senoidal() {
  return (sys.amplitudAuto * sin(2 * PI * timing.tiempo / timing.tiemporeferencia) + sys.offset);
}

float TriangularZonaMuerta() {
  float TiempoCiclo = fmod(timing.tiempo, timing.tiemporeferencia);
  float TiempoSeccion = timing.tiemporeferencia / 3.0;
  if (TiempoCiclo < TiempoSeccion)
    return (sys.amplitudAuto / TiempoSeccion) * TiempoCiclo;
  else if (TiempoCiclo < 2 * TiempoSeccion)
    return sys.amplitudAuto - ((sys.amplitudAuto / TiempoSeccion) * (TiempoCiclo - TiempoSeccion));
  else
    return 0;
}

float Impulso() {
  if (timing.tiempo < timing.tiemporeferencia)
    return sys.amplitudAuto + sys.offset;
  else
    return sys.offset;
}

float Chirp() {
  return (sys.amplitudAuto * sin(2 * PI * (0.001 * timing.tiempo) * timing.tiempo * 0.0001)) + sys.offset;
}

float DecaimientoExponencial() {
  return sys.amplitudAuto * exp(-0.0005 * timing.tiempo) + sys.offset;
}

float RuidoBlanco() {
  return (random(0, 100) / 100.0) * sys.amplitudAuto + sys.offset;
}

int GeneradorReferencia(int tiposenal) {
  switch (tiposenal) {
    case 0: return PRBS();
    case 1: return Cuadrada();
    case 2: return Senoidal();
    case 3: return TriangularZonaMuerta();
    case 4: return Impulso();
    case 5: return Chirp();
    case 6: return DecaimientoExponencial();
    case 7: return RuidoBlanco();
    default: return 0;
  }
  return 0;
}

inline void PID_update_coeficientes(float Ts) {
  static float Ts_prev = -1.f, Kd_prev = -1.f, Tc_prev = -1.f;
  const float Tc = fmaxf(ctrl.time_constant, 1e-6f);

  if (fabsf(Ts - Ts_prev) > 1e-9f || fabsf(ctrl.Kd - Kd_prev) > 1e-9f || fabsf(Tc - Tc_prev) > 1e-9f) {
    // D(s) = Kd*s/(1 + s*Tc)  ->  Tustin
    const float den = 2.f*Tc + Ts;
    ctrl.d_b0 =  (2.f*ctrl.Kd) / den;
    ctrl.d_b1 = -(2.f*ctrl.Kd) / den;
    ctrl.d_a1 =  (Ts - 2.f*Tc) / den;   // y[k] = -a1*y[k-1] + b0*e[k] + b1*e[k-1]
    Ts_prev   = Ts;  Kd_prev = ctrl.Kd;  Tc_prev = Tc;
  }
}

int PID_positional_Tustin() {
  float Ts = timing.tiempociclo / 1000.0f;
  PID_update_coeficientes(Ts);
  const float lo = (sys.modooperacion == 3) ? -255.0f : 0.0f;
  const float hi = 255.0f;

  // Error
  float e  = ctrl.error;
  float em = ctrl.previousError;

  // P
  float P = ctrl.Kp * e;

  // D via Tustin with first-order filter
  ctrl.d_y = -ctrl.d_a1 * ctrl.d_y + ctrl.d_b0 * e + ctrl.d_b1 * em;
  float D = ctrl.Kd >= 0 ? ctrl.d_y : 0.0f;   // safety, if you ever set Kd<0 by mistake

  // I via Tustin (trapezoidal) + back-calculation anti-windup
  float I_trial = ctrl.integral_sum + 0.5f * ctrl.Ki * Ts * (e + em);

  // Unsaturated output
  float u_unsat = P + I_trial + D;

  // Saturate
  float u_sat = constrain(u_unsat, lo, hi);

  // Anti-windup (tracking) on the integrator
  float Tt = fmaxf(ctrl.reset_time, 1e-6f);
  float I = I_trial + (Ts / Tt) * (u_sat - u_unsat);

  // Commit
  ctrl.integral_sum  = I;
  ctrl.controladorOUT = u_sat;
  ctrl.previousError  = e;

  return (int)(u_sat + 0.5f);
}

int PID_incremental() {
  float Ts = timing.tiempociclo / 1000.0f;
  PID_update_coeficientes(Ts);

  const float lo = (sys.modooperacion == 3) ? -255.0f : 0.0f;
  const float hi = 255.0f;

  // Δ términos
  float du_p = ctrl.Kp * (ctrl.error - ctrl.previousError);
  float du_i = 0.5f * ctrl.Ki * Ts * (ctrl.error + ctrl.previousError);

  // Derivada Tustin equivalente (usa el mismo filtro que el posicional)
  float d_out = -ctrl.d_a1 * ctrl.d_out_prev + ctrl.d_b0 * ctrl.error + ctrl.d_b1 * ctrl.previousError;
  float du_d  = d_out - ctrl.d_out_prev;
  ctrl.d_out_prev = d_out;

  float u_pre = ctrl.previousOutput + du_p + du_i + du_d;

  // Saturación + anti-windup por back-calculation (forma incremental)
  float u_sat_pre = constrain(u_pre, lo, hi);
  float Tt_eff = fmaxf(ctrl.reset_time, 1e-6f);
  float du_aw  = (u_sat_pre - u_pre) * (Ts / Tt_eff);

  float u = u_pre + du_aw;
  float u_sat = constrain(u, lo, hi);

  ctrl.controladorOUT = u_sat;
  ctrl.previousOutput = u_sat;
  ctrl.previousError  = ctrl.error;
  return (int)(u_sat + 0.5f);
}

int EcuacionDiferencias() { //Aquí, X es la entrada (error) y Y es la salida (PWM Arduino)
  ctrl.x[3] = ctrl.x[2];
  ctrl.x[2] = ctrl.x[1];
  ctrl.x[1] = ctrl.x[0];
  ctrl.x[0] = ctrl.error;
  ctrl.y[4] = ctrl.y[3];
  ctrl.y[3] = ctrl.y[2];
  ctrl.y[2] = ctrl.y[1];
  ctrl.y[1] = ctrl.controladorOUT;

  ctrl.controladorOUT = ctrl.A_cont * ctrl.x[0] + ctrl.B_cont * ctrl.x[1] + ctrl.C_cont * ctrl.x[2] + ctrl.D_cont * ctrl.x[3] +
                         ctrl.E_cont * ctrl.y[1] + ctrl.F_cont * ctrl.y[2] + ctrl.G_cont * ctrl.y[3] + ctrl.H_cont * ctrl.y[4];
  if (sys.modooperacion == 2)
    ctrl.controladorOUT = constrain(ctrl.controladorOUT, 0, 255);
  if (sys.modooperacion == 3)
    ctrl.controladorOUT = constrain(ctrl.controladorOUT, -255, 255);
  return ctrl.controladorOUT;
}

void setup() {
  Serial.setTimeout(5);
  pinMode(pins.pwmPin, OUTPUT);
  pinMode(pins.in1Pin, OUTPUT);
  pinMode(pins.in2Pin, OUTPUT);
  digitalWrite(pins.in1Pin, LOW);
  digitalWrite(pins.in2Pin, HIGH);
  pinMode(pins.encoderA, INPUT);
  pinMode(pins.encoderB, INPUT);
  attachInterrupt(digitalPinToInterrupt(pins.encoderA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pins.encoderB), updateEncoder, CHANGE);
  Serial.begin(115200);
  RPMAvg.begin();
  FilteredCurrent.begin();
  FilteredCurrent2.begin();
  randomSeed(analogRead(2));
}

void loop() {
  if (Serial.available() > 0) {
    serialData.DatoSerial = Serial.readStringUntil('\n');
    String parts[60];
    divideString(serialData.DatoSerial, ',', parts, 60);
    sys.inicio              = parts[0].toInt();
    sys.modooperacion       = parts[1].toInt();
    ctrl.A_cont             = parts[2].toFloat();
    ctrl.B_cont             = parts[3].toFloat();
    ctrl.C_cont             = parts[4].toFloat();
    ctrl.D_cont             = parts[5].toFloat();
    ctrl.E_cont             = parts[6].toFloat();
    ctrl.F_cont             = parts[7].toFloat();
    ctrl.G_cont             = parts[8].toFloat();
    ctrl.H_cont             = parts[9].toFloat();
    timing.delayintencional = parts[10].toInt();
    timing.tiemporeferencia = parts[11].toInt();
    sys.amplitudAuto        = parts[12].toInt();
    sys.referenciaManual    = parts[13].toInt();
    sys.offset              = parts[14].toInt();
    sys.tiposenal           = parts[15].toInt();
    sys.tab_activo          = parts[16].toInt();
    ctrl.Kp                 = parts[17].toFloat();
    ctrl.Ki                 = parts[18].toFloat();
    ctrl.Kd                 = parts[19].toFloat();
    ctrl.ZM                 = parts[20].toInt();
    ctrl.time_constant      = parts[21].toFloat();
    ctrl.PIDtype            = parts[22].toInt();
    sys.tiporef             = parts[23].toInt();
    ctrl.reset_time         = parts[24].toFloat();
    if (timing.delayintencional <= 1) timing.delayintencional = 1;
    inicializaVariables();
    timing.tiemporestart = micros() / 1000.0f;
  }

  timing.tiempoanterior      = timing.tiempo;
  timing.tiempo              = micros() / 1000.0f - timing.tiemporestart;
  timing.tiempoloop         = timing.tiempo-timing.tiempoanterior;
  timing.tiempociclo += timing.tiempoloop;

  if (timing.tiempociclo >= timing.delayintencional) { //Para bajar el jitter, se elimina el delay() e implementa esto
    if (sys.inicio == 1) {
      switch (sys.modooperacion) {
        case 0: // Inhabilitado
          analogWrite(pins.pwmPin, 0);
          delay(50);
          break;

        case 1: { // System Identification
          sys.referencia      = GeneradorReferencia(sys.tiposenal);
          ctrl.controladorOUT = sys.referencia;
          ctrl.controladorOUT = constrain(ctrl.controladorOUT, 0, 255);
          if (timing.tiempociclo < 1) timing.tiempociclo = timing.delayintencional;
          meas.RPM         = enc.encoderTicks / (float)timing.tiempociclo * 60.0f * 1000.0f / (4.0f * ctrl.PPR);
          meas.RPMPromedio = RPMAvg.reading(meas.RPM);
          enc.encoderTicks = 0;
          ctrl.PWMOUT = ctrl.controladorOUT;
          analogWrite(pins.pwmPin, ctrl.PWMOUT);
          sys.medicion = meas.RPMPromedio;
          break;
        }

        case 2: { // Control Velocidad
          sys.referencia = GeneradorReferencia(sys.tiposenal);
          if (timing.tiempociclo < 1) timing.tiempociclo = timing.delayintencional;
          meas.RPM         = enc.encoderTicks / (float)timing.tiempociclo * 60.0f * 1000.0f / (4.0f * ctrl.PPR);
          meas.RPMPromedio = RPMAvg.reading(meas.RPM);
          enc.encoderTicks = 0;

          ctrl.error = sys.referencia - meas.RPMPromedio;

          if (sys.tab_activo == 5){
            switch (ctrl.PIDtype) {
              case 0:  ctrl.controladorOUT = PID_incremental();       break;
              case 1:  ctrl.controladorOUT = PID_positional_Tustin(); break;
            }
            if (sys.referencia != 0){
              ctrl.PWMOUT = ctrl.controladorOUT + ctrl.ZM; // zona muerta solo si ref ≠ 0
              ctrl.PWMOUT = constrain(ctrl.PWMOUT, 0, 255);
            } else {
              ctrl.previousOutput = 0;
              ctrl.previousError  = 0;
              ctrl.controladorOUT = 0;
              ctrl.PWMOUT         = 0;
            }
          }
          else if (sys.tab_activo == 8){
            ctrl.controladorOUT = EcuacionDiferencias();
            if (sys.referencia != 0){
              ctrl.PWMOUT = ctrl.controladorOUT + ctrl.ZM;
              ctrl.PWMOUT = constrain(ctrl.PWMOUT, 0, 255);
            } else {
              ctrl.previousOutput = 0;
              ctrl.previousError  = 0;
              ctrl.controladorOUT = 0;
              ctrl.PWMOUT         = 0;
            }
          }
          analogWrite(pins.pwmPin, ctrl.PWMOUT);
          sys.medicion = meas.RPMPromedio;
          break;
        }

        case 3: { // Control Posición
          sys.referencia = GeneradorReferencia(sys.tiposenal);
          sys.deg        = enc.encoderTicks * (360.0f / (4.0f * ctrl.PPR));
          ctrl.error     = sys.referencia - sys.deg;
          if (timing.tiempociclo < 1) timing.tiempociclo = timing.delayintencional;

          float u = 0;
          if (sys.tab_activo == 5) {
            switch (ctrl.PIDtype) {
              case 0: u = PID_incremental();       break;
              case 1: u = PID_positional_Tustin(); break;
            }
          } else if (sys.tab_activo == 8) {
            u = EcuacionDiferencias();
          }
          u = constrain(u, -255, 255);

          #ifdef USE_ARDUINO
            if (u >= 0) {
              digitalWrite(pins.in1Pin, HIGH);
              digitalWrite(pins.in2Pin, LOW);
            } else {
              digitalWrite(pins.in1Pin, LOW);
              digitalWrite(pins.in2Pin, HIGH);
            }
          #endif

          #ifdef USE_TEENSY
            if (u >= 0) {
              digitalWrite(pins.in1Pin, LOW);   // DIR = forward
            } else {
              digitalWrite(pins.in1Pin, HIGH);  // DIR = reverse
            }
          #endif

          ctrl.controladorOUT = u;
          ctrl.PWMOUT         = abs((int)u); 
          analogWrite(pins.pwmPin, ctrl.PWMOUT);
          sys.medicion = (int)sys.deg;
          break;
        }
      }

      #ifdef USE_TEENSY
        meas.current = 1.26*analogRead(A5)+2.643; // ecuación de calibración
        meas.current = constrain(meas.current,0.0f,2000.0f);
        meas.currentMedian = FilteredCurrentMedian.reading(meas.current);
        meas.currentAvgAux = FilteredCurrent.reading(meas.currentMedian);
        meas.currentAvg = FilteredCurrent2.reading(meas.currentAvgAux);
        Serial.printf("%d %d %d %.2f %d\n",
                      (int)sys.referencia,
                      (int)sys.medicion,
                      (int)timing.tiempociclo,
                      (float)meas.currentAvg,   // no usado aún en la GUI
                      (int)ctrl.PWMOUT);
      #endif

      #ifdef USE_ARDUINO
        meas.current = analogRead(A5);
        Serial.print(sys.referencia);
        Serial.print(" ");
        Serial.print(int(sys.medicion));
        Serial.print(" ");
        Serial.print(int(timing.tiempociclo));
        Serial.print(" ");
        Serial.print((int)meas.current);
        Serial.print(" ");
        Serial.println(ctrl.PWMOUT);
      #endif
    } 
    else {
      analogWrite(pins.pwmPin, 0);
      delay(10);
    }
    timing.tiempociclo= 0;
  }
}
