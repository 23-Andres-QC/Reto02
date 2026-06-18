# Lógica de calibración, detección y rotación

Referencia única. Para cambiar comportamiento, modificar esto primero y reflejar el cambio aquí.

---

## Detección (`lane_detector.py`)

- **Amarillo y Blanco, ambos en HSV** (antes el blanco era LAB; se cambió porque el blanco real
  tiene baja saturación y alto brillo, más simple y consistente detectarlo así junto al amarillo)
- Centro del carril por banda: **C=(Y+W)/2 − white_bias** si se detectan ambos colores y la
  distancia entre ellos es razonable (60%-130% del ancho esperado); si solo hay uno de los dos,
  ese color ±11cm hacia el otro lado (amarillo+11cm o blanco-11cm − white_bias) — mitad de 22cm
  de carril. El carrito puede avanzar con CUALQUIERA de los dos colores, no solo amarillo.
  `white_bias_m` (2cm) desplaza el objetivo hacia el amarillo siempre que el blanco participa
  del cálculo (combinado o solo-blanco) — el robot se apegaba demasiado al blanco con el punto
  medio exacto; no se aplica al fallback solo-amarillo (ahí el blanco no participa)
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
- También se publica `/lane_slope`: la pendiente del **AMARILLO** (no el centro combinado),
  calculada **SOLO con los píxeles dentro de una franja ANGOSTA al fondo de la imagen**
  (`_inferior_slope`, ajusta una línea con `cv2.fitLine`) — NO usa la banda central ni la
  superior para nada. Los giros usan amarillo como referencia porque es la línea más confiable
  y continua en toda la pista (el blanco puede faltar o invalidarse en curvas). Dos parámetros
  separados controlan esto:
  - `slope_lookahead_m` (3cm por defecto): **DÓNDE se mide** — alto de la franja, en metros
    reales (no % de imagen, para no depender de la resolución de cámara). Chica = casi sin
    anticipar; muy chica = a veces se pierde el amarillo antes de comprometerse al giro
  - `slope_scale_m` (20cm por defecto): **CÓMO SE ESCALA** el valor reportado — se calcula la
    tangente real del ángulo (`dx/dy`, adimensional, NO depende de qué tan alta sea la franja
    de arriba) y se multiplica por esta distancia de referencia fija. Sin esto, achicar
    `slope_lookahead_m` para anticipar menos también achicaba `slope_m` para la MISMA curva
    real, y dejaba de cruzar los umbrales ya calibrados (`slope_curve_threshold`,
    `sharp_turn_slope_threshold`) — el robot "detectaba" la curva pero nunca llegaba a
    comprometerse al giro
  - Historial de ajuste de `slope_lookahead_m`: banda central vs inferior (versión original)
    anticipaba 7-10cm antes; toda la banda inferior (1/3 de imagen) ayudó pero seguía
    anticipando ~5cm; franja de 15% de imagen sin unidades físicas resultó demasiado angosta en
    la práctica (perdía el amarillo antes del giro); actual: 3cm reales, con `slope_scale_m`
    separado para no romper la calibración de los umbrales
  - **Bug de signo (corregido)**: al introducir `slope_scale_m` se devolvía `tangent *
    slope_scale_m` directamente. `tangent = vx/vy` representa dx/dy según "y" CRECE (hacia
    abajo en la imagen, hacia el robot), pero la convención histórica de `slope_m` era
    x_LEJOS − x_CERCA (la fórmula vieja evaluaba la línea en el borde lejano y en el cercano y
    restaba lejos−cerca). Eso equivale a `-tangent`, no a `+tangent` — el signo estaba
    invertido, así que a veces el giro salía hacia el lado contrario al real. Corregido a
    `-tangent * slope_scale_m`
  - **Explosión numérica (corregida)**: `tangent = vx/vy` puede crecer sin límite cuando la
    línea queda casi horizontal DENTRO de la franja (típico justo en medio de un giro cerrado),
    así que apenas un poco de ruido de unos píxeles producía valores de `slope_m` enormes y
    saltarines de un frame a otro — eso se sentía como giros bruscos y zigzag, y a veces el
    robot terminaba el giro mal alineado por haber girado de más/de menos en esos saltos. Se
    subió el piso de `|vy|` de 1e-6 (que en la práctica nunca se activaba) a `0.05`, y se agregó
    un tope final de ±0.35m sobre `slope_m` como respaldo adicional
- También se publica `/lane_error_yellow`: el error de centrado usando SOLO amarillo (amarillo +
  mitad del carril), calculado SIEMPRE que haya amarillo, sin importar si también hay blanco —
  a diferencia de `/lane_error`, que combina ambos colores cuando los dos están presentes y son
  válidos. `lane_controller.py` usa `/lane_error_yellow` específicamente MIENTRAS GIRA una
  esquina (ver sección "Esquina real" más abajo), porque durante el giro a veces aparece un
  blanco que pertenece a otro tramo de la pista y corrompe el error combinado
- Centroides pasan por filtro EMA antes de calcular error (reduce ruido frame a frame)
- **Zona de seguridad ANTICIPADA** (margen base 30% del carril ≈6.6cm desde cada línea, empujón ×1.2,
  agrandado por ángulo con ganancia 0.7, tope de error ±0.20m): si el robot se acerca demasiado a
  la amarilla o a la blanca, refuerza el error para alejarlo antes de cruzarla. El margen no es
  fijo: se AGRANDA según el ángulo actual (`slope_m`, calculado solo dentro de la banda
  inferior) — si la línea guía muestra que el carro se está angulando hacia una de las dos
  líneas, eso ya indica que va a salirse aunque todavía no esté dentro del margen estático,
  así que corrige antes (anticipado
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
   que restar). pos_x0,y0 (odom) se capturan en este momento, ni bien
   termina la espera. La IMU (yaw) solo se usa para el log de posición,
   NO en la ley de control (ver "Sin término de rumbo fijo" más abajo).
```

## Esquinas — SIN modo de giro dedicado

La pista tiene esquinas marcadas (en la práctica, cercanas a 90°), pero el
controlador **no tiene ningún modo especial para girar**. No hay `in_sharp_turn`,
no hay condición de salida, no hay ley de control separada para la esquina.
El giro lo ejecuta `lane_detector.py` simplemente cambiando a qué le presta
atención: cuando el blanco desaparece de cuadro (entrando a la esquina), el
error pasa de "centro entre amarillo y blanco" a "solo amarillo, con su
fallback ±11cm" (ver Detección) — ese salto de referencia es lo que hace que
el error crezca y el PID normal gire fuerte, sin que el controlador necesite
saber que está en una curva. Al reaparecer el blanco del tramo siguiente, el
error vuelve a ser el combinado normal y el PID sigue centrando como siempre.

**Por qué se eliminó el modo dedicado (`in_sharp_turn`)**: una versión anterior
tenía un bloque separado, activado por umbral de pendiente (`/lane_slope`),
con su propia ley de control, su propia condición de salida (doble: amarillo
+ combinado) y su propio suavizado. Esa arquitectura acumuló, en sucesivas
vueltas de ajuste, un bug de signo en el cálculo de pendiente, explosión
numérica cuando la línea quedaba casi horizontal, retraso de varios cientos
de ms por el suavizado que producía sobregiro, y una condición de salida que
casi nunca se cumplía a tiempo (causando zigzag al volver a avanzar). Quitar
el modo dedicado por completo y dejar que la misma ley PID+FF continua
maneje todo —recta y curva— elimina esa categoría entera de bugs: no hay
condición de salida que falle porque no hay nada que "salir".

`/lane_slope` y `/lane_error_yellow` siguen existiendo en `lane_detector.py`
(sin tocar) — `/lane_slope` todavía alimenta el feed-forward (FF) de
anticipación en la ley PID normal (ver más abajo); `/lane_error_yellow` ya no
lo consume nadie, queda publicado pero sin uso en el control real.

## Control en avance — una sola ley PID, SIEMPRE (recta y curva)

```
e = error (banda inferior, ver Detección)  (si |e|<1cm → 0, ruido)
P = kp*e   I = ki*integral(e), anti-windup   D = kd*(e-e_anterior)/dt

anticipa_curva = |slope| > slope_curve_threshold (4cm)
  → PREDICTIVO: usa la pendiente del amarillo calculada SOLO dentro de la banda
    INFERIOR (no la central ni la superior — esas miran más adelante en la
    pista de lo que el robot ya alcanzó, anticipando demasiado pronto), no el
    giro que el robot ya está haciendo.

FF = kff*slope                → solo si anticipa_curva (slope = lectura del frame
                                 actual, sin retraso; "tendencia" del error tardaba
                                 ~0.5s en acumularse y llegaba tarde a la curva)
ω = -(P+I+D+FF), limitado a max_angular, suavizado (alpha=0.12)
v = linear_speed * curve_speed_factor (0.30 × 0.6 = 0.18 m/s) — SIEMPRE, no solo en curvas
```

`turn_threshold`/`last_w` quedaron obsoletos y se eliminaron — la anticipación de curva
ahora es predictiva (vía cámara/slope) en vez de reactiva (vía el propio giro del robot).

**Sin término de rumbo fijo (`yaw_term`, eliminado).** Existía un empujón hacia un
`initial_yaw` capturado una sola vez al arrancar, pensado como ayuda contra deriva
lenta en tramos rectos. Se quitó porque, sin recalibrarse nunca, ese rumbo "objetivo"
seguía apuntando a la dirección de ANTES de cada esquina real — el carril ya había
girado ~90° pero el término seguía empujando hacia el rumbo viejo, compitiendo con
esta misma ley de control (que sí ve la nueva dirección vía `e`/`slope`) y produciendo
zigzag sostenido después de cada giro. El control depende 100% de lo que la cámara
ve en cada frame, nunca de un plan de rumbo fijo.

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
4. Centro objetivo: **C=(Y+W)/2 − white_bias (2cm)** si hay ambos colores; si solo uno, ese
   color ±11cm hacia el otro lado (amarillo+11cm o blanco-11cm − white_bias) — nunca el centro
   de la imagen. El error de control usa SOLO la banda inferior (la posición actual real), no
   un promedio de las 3 bandas
5. Una sola ley de control PID — no ramas con ganancias distintas según error/tendencia
6. No agregar una ganancia de proximidad CONTINUA (causó zigzag) — la única excepción permitida
   es la zona de seguridad discreta (solo dentro de 25% del carril desde cada línea), que existe
   específicamente para no salirse del carril, no para "ayudar" al centrado general
7. `self.error` nunca se pisa con `None` en NaN — solo `age` decide el estado
8. Al arrancar NO hay calibración activa (no gira en el sitio) — solo espera quieto start_delay
   y luego avanza con PID directo desde la posición real en la que está el robot
9. No usar ningún rumbo/objetivo fijo capturado una sola vez (yaw inicial, ángulo planeado,
   etc.) en la ley de control — no sigue al carril después de una esquina real y compite
   con la corrección visual, causando zigzag. Todo input de control viene del frame actual
   (e, slope); la IMU solo se usa para diagnóstico/log
10. `lane_width_m` debe ser **0.22** (22cm reales) para que la mitad sea exactamente 11cm — si se
    cambia, el log de separación se recalcula solo (usa `target_cm` derivado, no un número fijo)
11. NO reintroducir un modo de giro dedicado (`in_sharp_turn` o similar) en `lane_controller.py`.
    El giro lo ejecuta el cambio de referencia de `lane_detector.py` (amarillo solo, fallback
    ±11cm) cuando el blanco desaparece — la misma ley PID+FF de siempre lo maneja, sin saber
    que está en una curva. Un modo dedicado anterior acumuló bugs de signo, explosión numérica
    y condiciones de salida que nunca se cumplían a tiempo
