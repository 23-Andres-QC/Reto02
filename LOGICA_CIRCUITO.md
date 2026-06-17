# Lógica de calibración, detección y rotación

Referencia única. Para cambiar comportamiento, modificar esto primero y reflejar el cambio aquí.

---

## Detección (`lane_detector.py`)

- **Amarillo y Blanco, ambos en HSV** (antes el blanco era LAB; se cambió porque el blanco real
  tiene baja saturación y alto brillo, más simple y consistente detectarlo así junto al amarillo)
- Centro del carril por banda: **C=(Y+W)/2** si se detectan ambos colores y la distancia entre
  ellos es razonable (60%-130% del ancho esperado); si solo hay uno de los dos, ese color ±11cm
  hacia el otro lado (amarillo+11cm o blanco-11cm) — mitad de 22cm de carril. El carrito puede
  avanzar con CUALQUIERA de los dos colores, no solo amarillo
- Filtro de forma: **elongación por PCA** sobre componentes conectados (`_component_filter`),
  aplicado a ambos colores — más robusto que un bounding-box para distinguir cintas largas de
  manchas/reflejos redondeados. Reemplazó el filtro anterior (área + aspecto de bounding-box)
- Sin amarillo NI blanco → NaN (el controlador frena por completo, ver sección de frenado)
- **3 bandas horizontales fijas** (superior, central, inferior — cada una 1/3 de la imagen): se mide
  el centro en cada una. El error final (`/lane_error`, lo que usa el controlador como "dónde está
  el robot") usa **solo la banda INFERIOR** — es la más cercana al robot, la que de verdad indica su
  posición lateral actual. Las bandas superior y central NO son "dónde está el robot": son la pista
  más adelante, y solo alimentan la pendiente (`/lane_slope`, ver abajo) para anticipar curvas.
  (Antes el error promediaba las 3 bandas — mezclaba la posición actual con la guía futura.)
- Esos 3 puntos se recalculan cada frame y trazan la **línea de recorrido (guía)** que se dibuja en
  el debug — no es una línea fija, se vuelve a trazar constantemente según dónde esté el amarillo
- También se publica `/lane_slope`: la pendiente del **AMARILLO** (no el centro combinado) entre
  su punto **central** e **inferior** (0 = recta/vertical, distinto de 0 = el robot está angulado
  respecto a la pista). Los giros usan amarillo como referencia porque es la línea más confiable y
  continua en toda la pista (el blanco puede faltar o invalidarse en curvas). Se usa centro-inferior
  y NO superior-inferior — el punto superior (banda lejana) ve la curva mucho antes de que el robot
  realmente llegue, así que usarlo anticipaba el giro demasiado pronto. Se usa en la calibración
  inicial para exigir que el robot arranque realmente derecho, no solo centrado
- Centroides pasan por filtro EMA antes de calcular error (reduce ruido frame a frame)
- **Zona de seguridad ANTICIPADA** (margen base 30% del carril ≈6.6cm desde cada línea, empujón ×1.2,
  agrandado por ángulo con ganancia 0.7, tope de error ±0.20m): si el robot se acerca demasiado a
  la amarilla o a la blanca, refuerza el error para alejarlo antes de cruzarla. El margen no es
  fijo: se AGRANDA según el ángulo actual (`slope_m`, centro vs inferior) — si la línea guía
  muestra que el carro se está angulando hacia una de las dos líneas, eso ya indica que va a
  salirse aunque todavía no esté dentro del margen estático, así que corrige antes (anticipado
  por ángulo), no solo cuando ya está encima de la línea (reactivo por posición). Las ganancias
  se bajaron (antes ×1.8/1.2) y se agregó el tope de error porque en curvas este empujón se
  sumaba al PID + FF y componía una corrección demasiado fuerte (el error rebotaba de un extremo
  a otro, ej. -25cm a +5cm, en vez de asentarse suave). Sigue siendo un empujón discreto, no una
  ganancia continua duplicada con el PID (eso causó zigzag en una versión anterior, ver Reglas de
  oro #6)
- Debug (`/lane/debug_image`): overlay translúcido sobre la cámara real (bird's-eye), no fondo negro;
  **3 líneas verdes** = las 3 bandas de medición; **línea magenta** = recorrido planeado (3 puntos)

## Error y signos

```
error > 0 → robot desplazado a la IZQUIERDA → girar DERECHA → ω < 0
error < 0 → robot desplazado a la DERECHA   → girar IZQUIERDA → ω > 0
```

## Arranque (`lane_controller.py`)

```
1. ESPERA: al detectar amarillo y/o blanco por primera vez, arranca el
   cronómetro de start_delay (5s) y se queda quieto (no avanza, no gira
   en el sitio) — SIN calibración activa. (Antes había una fase de
   calibración activa que giraba en el sitio ajustando ángulo antes de
   avanzar; se quitó porque la corrección resultaba muy brusca y sesgaba
   el giro hacia la derecha. Ahora solo se detecta la pista y se espera.)

2. AVANCE: termina la espera (start_delay) → PID normal directo, usando
   la posición real en la que esté el robot (no hay bias de calibración
   que restar). initial_yaw (IMU) y pos_x0,y0 (odom) se capturan en este
   momento, ni bien termina la espera.
```

## Esquina ~90° (track de curvas reales, no continuas)

La pista tiene esquinas de ~90°, no curvas suaves continuas. Si la pendiente crece
mucho (la línea se va casi de canto), no es algo para corregir con FF — es una
esquina real. Se activa un modo dedicado, ANTES de la ley PID normal:

```
Dos formas de entrar en in_sharp_turn = True:
  1. Por MAGNITUD: |slope| > sharp_turn_slope_threshold (13cm, antes 9cm — se
     disparaba 7-10cm antes de tiempo, demasiado temprano para una esquina real)
  2. Por TIEMPO: lleva anticipando (|slope| > slope_curve_threshold, 4cm) de forma
     continua más de max_anticipation_time (0.8s) sin resolver → fuerza el giro
     cerrado igual, aunque la magnitud no haya llegado al umbral de esquina.
     (Antes el robot podía quedarse anticipando con el FF suave hasta ~2s antes
     de comprometerse al giro real — se sentía como un giro adelantado/abierto.
     Este tope de tiempo limita esa ventana.)

Mientras in_sharp_turn:
  dirección = signo de slope (mismo signo que la corrección normal)
  w = -(sharp_turn_w * dirección + sharp_turn_kp_e * e), limitado a ±sharp_turn_max_w
  angular.z = suavizado (alpha=0.10)
  linear.x  = v * sharp_turn_speed_factor (0.30 × 0.3 = 0.09 m/s) — muy reducida
  Sale del giro (in_sharp_turn = False) cuando:
    |slope| < slope_curve_threshold (4cm)  Y  |e| < calib_tolerance (2.5cm)
    → la línea ya está recta y centrada otra vez (la "siguiente" línea tras la esquina)

El giro NO es a un ritmo fijo: se suma una corrección proporcional al error
lateral ACTUAL (sharp_turn_kp_e * e). Antes, una velocidad angular fija
generaba un giro de radio constante ("abierto") que no necesariamente
converge al centro real — si el robot llegaba a la esquina ya desviado, el
giro fijo no compensaba eso y se acababa saliendo de la línea amarilla en
vez de seguirla. Con la corrección proporcional, si llega desviado, el giro
se cierra más para converger hacia el centro de las líneas proyectadas.
```

Mientras `in_sharp_turn=True`, el control_loop sale antes de llegar al PID normal
(la ley PID de avance no se ejecuta esos frames).

## Control en avance — una sola ley PID

```
e = error (banda inferior, ver Detección)  (si |e|<1cm → 0, ruido)
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
estar alineado. La única excepción real es perder TODO color (ver sección siguiente).

## Sin color (ni amarillo ni blanco): FRENA por completo

Regla simple y explícita: el carrito **solo avanza si detecta amarillo O blanco**
(cualquiera de los dos, el detector ya hace el fallback correspondiente). Si no
detecta NINGUNO de los dos colores, frena por completo — no avanza, no gira
buscando — y se queda quieto hasta volver a detectar cualquiera de los dos.

```
age = tiempo desde la última lectura válida (amarillo o blanco)

age <= error_timeout (0.5s)  → PID normal (avanza)
age > error_timeout          → FRENA por completo: linear.x=0, angular.z=0
                                 resetea integral, in_sharp_turn, anticipation_timer
                                 en cuanto vuelve a detectar, retoma el PID de inmediato
```

Esto reemplazó el sistema anterior (pérdida breve congelada + búsqueda activa girando
mientras avanzaba) — se simplificó a una regla binaria: hay color → avanza, no hay
color → frena. No hay modo de "búsqueda" que mueva el robot a ciegas.

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

1. El carrito SOLO avanza si detecta amarillo o blanco — si no detecta ninguno, FRENA por
   completo (no hay modo de "búsqueda" que mueva el robot a ciegas sin ver ningún color)
2. En cuanto vuelve a detectar cualquiera de los dos colores, retoma el PID normal de inmediato
3. Mientras SÍ detecta color, la corrección siempre es gradual avanzando — nunca "frenar para
   corregir" (eso es distinto de frenar por falta total de color, regla #1)
4. Centro objetivo: **C=(Y+W)/2** si hay ambos colores; si solo uno, ese color ±11cm hacia el
   otro lado (amarillo+11cm o blanco-11cm) — nunca el centro de la imagen. El error de control
   usa SOLO la banda inferior (la posición actual real), no un promedio de las 3 bandas
5. Una sola ley de control PID — no ramas con ganancias distintas según error/tendencia
6. No agregar una ganancia de proximidad CONTINUA (causó zigzag) — la única excepción permitida
   es la zona de seguridad discreta (solo dentro de 25% del carril desde cada línea), que existe
   específicamente para no salirse del carril, no para "ayudar" al centrado general
7. `self.error` nunca se pisa con `None` en NaN — solo `age` decide el estado
8. Al arrancar NO hay calibración activa (no gira en el sitio) — solo espera quieto start_delay
   y luego avanza con PID directo desde la posición real en la que está el robot
9. El yaw planeado es un empujón suave, no una línea rígida — se apaga en curvas reales
10. `lane_width_m` debe ser **0.22** (22cm reales) para que la mitad sea exactamente 11cm — si se
    cambia, el log de separación se recalcula solo (usa `target_cm` derivado, no un número fijo)
11. La pista tiene esquinas ~90° reales, no curvas suaves continuas — por eso existen dos umbrales
    de `slope` distintos: uno para anticipar (FF, suave) y otro mayor para esquina real (giro
    lento dedicado, `in_sharp_turn`) — no confundirlos ni unificarlos en uno solo
