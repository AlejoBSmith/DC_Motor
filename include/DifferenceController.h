#pragma once
#include <math.h>

// b = [A, B, C, D, I], feedback = [E, F, G, H] = -a[1..4].
// The state stores the saturated controller output, before PWM quantization
// and before speed-mode dead-zone compensation.
struct DifferenceController {
  float b[5] = {};
  float feedback[4] = {};
  float x[5] = {};
  float y[4] = {};

  void reset() {
    for (int i = 0; i < 5; ++i) x[i] = 0.0f;
    for (int i = 0; i < 4; ++i) y[i] = 0.0f;
  }

  float step(float error, float lo, float hi) {
    for (int i = 4; i > 0; --i) x[i] = x[i - 1];
    x[0] = error;
    float output = 0.0f;
    for (int i = 0; i < 5; ++i) output += b[i] * x[i];
    for (int i = 0; i < 4; ++i) output += feedback[i] * y[i];
    output = fminf(fmaxf(output, lo), hi);
    for (int i = 3; i > 0; --i) y[i] = y[i - 1];
    y[0] = output;
    return output;
  }
};

inline int pwmFullScale(int bits) {
  bits = bits < 1 ? 1 : (bits > 16 ? 16 : bits);
  return (int)((1UL << bits) - 1UL);
}

inline int pwmFromNormalized(float output, int bits = 8) {
  const float normalized = fminf(fmaxf(output, 0.0f), 255.0f);
  return (int)lroundf(normalized * (pwmFullScale(bits) / 255.0f));
}

inline float pwmToNormalized(int counts, int bits = 8) {
  return counts * (255.0f / pwmFullScale(bits));
}

inline int speedPwm(float output, int deadzone, int bits = 8) {
  return pwmFromNormalized(output + deadzone, bits);
}

inline int positionPwm(float output, int bits = 8) {
  return pwmFromNormalized(fabsf(output), bits);
}
