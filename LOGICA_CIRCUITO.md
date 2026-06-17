# Lógica de calibración, detección y rotación

Referencia única. Para cambiar comportamiento, modificar esto primero y reflejar el cambio aquí.

---

## Detección (`lane_detector.py`)

- **Amarillo y Blanco, ambos en HSV** (antes el blanco era LAB; se cambió porque el blanco real
  tiene baja saturación y alto brillo, más simple y consistente detectarlo así junto al amarillo)
- Centro del carril por banda: **C=(Y+W)/2** si se detectan ambos colores y la distancia entre
  ellos es razonable (60%-130% del ancho esperado); si solo hay amarillo, centro = amarillo + 11cm
  (mitad de 22cm de carril)
- Filtro de forma: **elongación por PCA** sobre componentes conectados (`_component_filter`),
  aplicado a ambos colores — más robusto que un bounding-box para distinguir cintas largas de
  manchas/reflejos redondeados. Reemplazó el filtro anterior (área + aspecto de bounding-box)
- Sin amarillo → NaN (entra en búsqueda)
- **3 bandas horizontales fijas** (superior, central, inferior — cada una 1/3 de la imagen): se mide
  el centro en cada una y el error final usa el **promedio de las 3** — aprovecha toda la línea
  visible, no un solo punto
- Esos 3 puntos se recalculan cada frame y trazan la **línea de recorrido (guía)** que el robot intenta
  minimizar de error para avanzar recto — no es una línea fija, se vuelve a trazar constantemente
  según dónde esté el amarillo
- También se publica `/lane_slope`: la pendiente del **AMARILLO** (no el centro combinado) entre
  su punto **central** e **inferior** (0 = recta/vertical, distinto de 0 = el robot está angulado
  respecto a la pista). Los giros usan amarillo como referencia porque es la línea más confiable y
  continua en toda la pista (el blanco puede faltar o invalidarse en curvas). Se usa centro-inferior
  y NO superior-inferior — el punto superior (banda lejana) ve la curva mucho antes de que el robot
  realmente llegue, así que usarlo anticipaba el giro demasiado pronto. Se usa en la calibración
  inicial para exigir que el robot arranque realmente derecho, no solo centrado
- Centroides pasan por filtro EMA antes de calcular error (reduce ruido frame a frame)
- **Zona de seguridad ANTICIPADA** (margen base 30% del carril ≈6.6cm desde cada línea, empujón ×1.8):
  si el robot se acerca demasiado a la amarilla o a la blanca, refuerza el error para alejarlo
  antes de cruzarla. El margen no es fijo: se AGRANDA según el ángulo actual (`slope_m`, centro
  vs inferior) — si la línea guía muestra que el carro se está angulando hacia una de las dos
  líneas, eso ya indica que va a salirse aunque todavía no esté dentro del margen estático, así
  que corrige antes (anticipado por ángulo), no solo cuando ya está encima de la línea (reactivo
  por posición). Sigue siendo un empujón discreto, no una ganancia continua duplicada con el PID
  (eso causó zigzag en una versión anterior, ver Reglas de oro #6)
- Debug (`/lane/debug_image`): overlay translúcido sobre la cámara real (bird's-eye), no fondo negro;
  **3 líneas verdes** = las 3 bandas de medición; **línea magenta** = recorrido planeado (3 puntos)

## Error y signos

```
error > 0 → robot desplazado a la IZQUIERDA → girar DERECHA → ω < 0
error < 0 → robot desplazado a la DERECHA   → girar IZQUIERDA → ω > 0
```

## Arranque (`lane_controller.py`)

```
1. CALIBRACIÓN ACTIVA (no avanza, solo ajusta ángulo):
   se activa al ver amarillo por primera vez.
   e     = error promedio (centrado lateral)
   slope = pendiente del amarillo (banda central vs inferior, /lane_slope) — 0 = recta,
           distinto de 0 = el robot está angulado respecto a la pista aunque
           el centrado promedio ya esté bien
   w = -(calib_kp * e + calib_kp_slope * slope)
   Si hace falta corregir pero |w| < calib_min_w (0.12 rad/s), se sube a ese
   piso mínimo conservando el signo — un comando muy chico puede no superar
   la zona muerta/fricción del motor real y el robot se queda sin moverse
   aunque el controlador sí esté calculando una corrección.
   limitado a ±calib_w
   Calibrado cuando |e| y |tendencia(e)| < calib_tolerance (2.5cm)
   Y |slope| < slope_tolerance (3cm)
   durante calib_stable_frames (8 frames ≈0.25s) seguidos — así no solo queda
   bien centrado en promedio, sino realmente derecho respecto a la pista.
   (Valores relajados tras observar en pista real que 1.2cm/15 frames era
   demasiado estricto: el error oscila por ruido/deriva física y nunca
   llegaba a juntar suficientes frames seguidos — el robot se quedaba
   atascado en esta fase sin avanzar nunca.)

2. ESPERA: calibrado → captura calib_bias=e, initial_yaw (IMU),
   pos_x0,y0 (odom) → espera start_delay (5s) sin moverse.

3. AVANCE: termina la espera → PID normal.
```

## Esquina ~90° (track de curvas reales, no continuas)

La pista tiene esquinas de ~90°, no curvas suaves continuas. Si la pendiente crece
mucho (la línea se va casi de canto), no es algo para corregir con FF — es una
esquina real. Se activa un modo dedicado, ANTES de la ley PID normal:

```
Si |slope| > sharp_turn_slope_threshold (9cm)  → entra en in_sharp_turn = True

Mientras in_sharp_turn:
  dirección = signo de slope (mismo signo que la corrección normal)
  angular.z = giro lento dedicado (sharp_turn_w = 0.40 rad/s), suavizado (alpha=0.10)
  linear.x  = v * sharp_turn_speed_factor (0.30 × 0.3 = 0.09 m/s) — muy reducida
  Sale del giro (in_sharp_turn = False) cuando:
    |slope| < slope_curve_threshold (4cm)  Y  |e| < calib_tolerance (2.5cm)
    → la línea ya está recta y centrada otra vez (la "siguiente" línea tras la esquina)
```

Mientras `in_sharp_turn=True`, el control_loop sale antes de llegar al PID normal
(la ley PID de avance no se ejecuta esos frames).

## Control en avance — una sola ley PID

```
e = error - calib_bias  (si |e|<1cm → 0, ruido)
P = kp*e   I = ki*integral(e), anti-windup   D = kd*(e-e_anterior)/dt

anticipa_curva = |slope| > slope_curve_threshold (4cm)
  → PREDICTIVO: usa la pendiente de la línea guía (banda CENTRAL vs INFERIOR,
    no superior vs inferior — el punto lejano anticipaba demasiado pronto),
    no el giro que el robot ya está haciendo.

FF = kff*slope                → solo si anticipa_curva (slope = lectura del frame
                                 actual, sin retraso; "tendencia" del error tardaba
                                 ~0.5s en acumularse y llegaba tarde a la curva)
yaw_term = yaw_correction * yaw_weight(0.3)  → solo si NO anticipa_curva
ω = -(P+I+D+FF) + yaw_term, limitado a max_angular, suavizado (alpha=0.12)
v = linear_speed * curve_speed_factor (0.30 × 0.6 = 0.18 m/s) — SIEMPRE, no solo en curvas
```

`turn_threshold`/`last_w` quedaron obsoletos y se eliminaron — la anticipación de curva
ahora es predictiva (vía cámara/slope) en vez de reactiva (vía el propio giro del robot).

`yaw_term` no es una línea rígida: es solo un empujón suave que evita deriva lenta
en tramos rectos. En curvas reales se apaga (igual que FF) para no resistir el giro
ya anticipado por la cámara.

**La calibración durante el avance nunca es "frenar y girar"**: el robot siempre sigue
avanzando mientras `angular.z` se ajusta gradualmente hacia la línea de recorrido
recalculada cada frame. Si se desvía un poco, no se detiene a corregir — simplemente
avanza con un pequeño sesgo angular hacia donde está la línea guía, hasta volver a
estar alineado. Nunca cambia a "parar y rotar" (eso solo pasa en pérdida sostenida
de amarillo, ver sección siguiente).

## Pérdida de amarillo — dos umbrales (evita zigzag)

```
age = tiempo desde última lectura válida

age <= error_timeout (0.8s)              → PID normal
0.8s < age <= search_timeout (2.2s)      → pérdida breve: usa el e CONGELADO,
                                             sin tocar integral/derivada, sigue recto
age > search_timeout (2.2s)              → búsqueda real: gira izquierda (drift_w)
                                             con rampa lenta (alpha=0.06) + corrección
                                             hacia yaw inicial, avanza al 40%, nunca para
```

## Esquinas

Acumula ángulo girado en una dirección. Si supera 90° → `in_corner=True` (informativo, no cambia la ley de control).

## Herramienta de calibración HSV offline (no ROS)

`tools/preprocesamiento_lineas_hsv.py` — script standalone (no nodo ROS) para calibrar
visualmente los rangos HSV de amarillo/blanco a partir de una imagen guardada (captura de
rqt_image_view o frame de `/image_raw`). Uso: `python3 tools/preprocesamiento_lineas_hsv.py captura.png`.
Genera 4 imágenes de prueba progresivas (HSV simple → ROI+morfología → componentes filtrados →
resultado final con Hough + centro) más las máscaras finales. Los rangos HSV de blanco/amarillo
y el filtro de elongación PCA usados en `lane_detector.py` salieron de experimentar con este script.

## Herramienta de calibración (sin mover el robot, nodo ROS)

`capytown_esan/calib_hsv_lab.py` — réplica exacta de la detección de `lane_detector.py`
(HSV para ambos colores, filtro PCA, 3 bandas, línea guía, error, separación) pero
**nunca publica `/cmd_vel`** y
no depende de `lane_controller`. Sirve para mover el robot a mano y ver en consola/debug
qué calcula en cada posición/ángulo, sin riesgo de que se mueva.

```bash
ros2 run capytown_esan calib_hsv_lab
```

Requiere que la cámara esté publicando en `/image_raw` (mismo topic que usa `lane_detector`).
Debug visual en `/calib/debug_image` (ver con `rqt_image_view`). Imprime cada ~0.3s:
amarillo (px), separación (cm), error (cm), yaw (IMU), posición (odometría), puntos de la línea guía.

## Reglas de oro

1. Nunca frenar del todo (salvo calibración inicial / espera) — siempre avanza, aunque sea despacio
2. Pérdida sostenida → buscar izquierda con rampa lenta, nunca salto brusco a velocidad fija
3. Pérdida breve → usar última lectura congelada, no cambiar de modo
4. Centro objetivo: **C=(Y+W)/2** si hay ambos colores, si no, **amarillo + 11cm** — nunca el centro de la imagen
5. Una sola ley de control PID — no ramas con ganancias distintas según error/tendencia
6. No agregar una ganancia de proximidad CONTINUA (causó zigzag) — la única excepción permitida
   es la zona de seguridad discreta (solo dentro de 25% del carril desde cada línea), que existe
   específicamente para no salirse del carril, no para "ayudar" al centrado general
7. `self.error` nunca se pisa con `None` en NaN — solo `age` decide el estado
8. Al arrancar, calibrar activamente (ángulo) antes de la espera — nunca asumir que ya está bien puesto
9. El yaw planeado es un empujón suave, no una línea rígida — se apaga en curvas reales
10. `lane_width_m` debe ser **0.22** (22cm reales) para que la mitad sea exactamente 11cm — si se
    cambia, el log de separación se recalcula solo (usa `target_cm` derivado, no un número fijo)
11. La pista tiene esquinas ~90° reales, no curvas suaves continuas — por eso existen dos umbrales
    de `slope` distintos: uno para anticipar (FF, suave) y otro mayor para esquina real (giro
    lento dedicado, `in_sharp_turn`) — no confundirlos ni unificarlos en uno solo
