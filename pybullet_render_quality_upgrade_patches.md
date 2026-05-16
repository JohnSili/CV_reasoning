# PyBullet Quality Upgrade Patches

Ниже — готовые патчи для:

- Temporal AA
- ACES tonemapping
- randomized lighting
- camera jitter
- sensor simulation
- grain
- atmospheric haze
- better realism

Все куски можно вставлять напрямую.

---

# 1. IMPORTS

Добавь:

```python
import math
```

---

# 2. ЗАМЕНИ `postprocess()`

```python
def postprocess(img: np.ndarray) -> np.ndarray:
    pil = PILImage.fromarray(img)

    # slight cinematic blur
    pil = pil.filter(
        ImageFilter.GaussianBlur(radius=0.35)
    )

    # soft sharpen
    pil = pil.filter(
        ImageFilter.UnsharpMask(
            radius=0.8,
            percent=70,
            threshold=3,
        )
    )

    # mild grading
    pil = ImageEnhance.Contrast(pil).enhance(1.05)
    pil = ImageEnhance.Color(pil).enhance(1.06)
    pil = ImageEnhance.Brightness(pil).enhance(1.04)

    arr = np.array(pil, dtype=np.float32) / 255.0

    # -------------------------------------------------
    # FILMIC ACES TONEMAP
    # -------------------------------------------------

    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    arr = np.clip(
        (arr * (a * arr + b)) /
        (arr * (c * arr + d) + e),
        0,
        1,
    )

    # slight gamma
    arr = np.power(arr, 0.98)

    h, w = arr.shape[:2]

    # -------------------------------------------------
    # ATMOSPHERIC HAZE
    # -------------------------------------------------

    fog = np.linspace(1.0, 0.94, h)[:, None, None]
    arr *= fog

    # -------------------------------------------------
    # VIGNETTE
    # -------------------------------------------------

    Y, X = np.ogrid[:h, :w]

    dist = np.sqrt(
        ((X - w/2) / (w/2))**2 +
        ((Y - h/2) / (h/2))**2
    )

    vignette = 1.0 - np.clip(dist * 0.10, 0, 0.06)

    arr *= vignette[:, :, None]

    # -------------------------------------------------
    # SENSOR GRAIN
    # -------------------------------------------------

    grain = np.random.normal(0, 0.004, arr.shape)
    arr = np.clip(arr + grain, 0, 1)

    # -------------------------------------------------
    # WOOD / TEXTURE VARIATION
    # -------------------------------------------------

    texture_noise = np.random.normal(1.0, 0.01, (h, w))
    arr *= texture_noise[:, :, None]

    arr = np.clip(arr, 0, 1)

    return (arr * 255).astype(np.uint8)
```

---

# 3. ЗАМЕНИ `_render()`

```python
def _render(self, yaw, pitch, distance):

    distance = max(0.7, min(distance, 1.8))
    pitch = max(-80, min(pitch, -10))

    W = self.width * self.SSAA
    H = self.height * self.SSAA

    target = [0, 0, self.TABLE_HEIGHT + self.TABLE_THICKNESS / 2]

    # -------------------------------------------------
    # RANDOMIZED CAMERA
    # -------------------------------------------------

    yaw += np.random.normal(0, 1.0)
    pitch += np.random.normal(0, 0.8)
    distance += np.random.normal(0, 0.025)

    proj = p.computeProjectionMatrixFOV(
        fov=44,
        aspect=W / H,
        nearVal=0.01,
        farVal=10.0,
    )

    # -------------------------------------------------
    # TEMPORAL AA
    # -------------------------------------------------

    samples = []

    TAA_SAMPLES = 4

    for _ in range(TAA_SAMPLES):

        jitter_yaw = yaw + np.random.normal(0, 0.08)
        jitter_pitch = pitch + np.random.normal(0, 0.08)

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=jitter_yaw,
            pitch=jitter_pitch,
            roll=0,
            upAxisIndex=2,
        )

        # -------------------------------------------------
        # RANDOMIZED LIGHTING
        # -------------------------------------------------

        key_strength = np.random.uniform(0.96, 1.04)
        fill_strength = np.random.uniform(0.96, 1.04)

        key = self._render_pass(
            W,
            H,
            view,
            proj,
            light_dir=[
                0.55 + np.random.normal(0, 0.02),
                0.35 + np.random.normal(0, 0.02),
                0.72,
            ],
            light_color=[
                1.00 * key_strength,
                0.97 * key_strength,
                0.92 * key_strength,
            ],
            ambient=0.28,
            diffuse=0.72,
            specular=0.16,
            shadow=True,
        )

        fill = self._render_pass(
            W,
            H,
            view,
            proj,
            light_dir=[
                -0.65,
                0.25,
                0.45,
            ],
            light_color=[
                0.90 * fill_strength,
                0.93 * fill_strength,
                1.00 * fill_strength,
            ],
            ambient=0.22,
            diffuse=0.28,
            specular=0.02,
            shadow=False,
        )

        rim = self._render_pass(
            W,
            H,
            view,
            proj,
            light_dir=[-0.2, -1.0, 0.45],
            light_color=[0.96, 0.96, 1.00],
            ambient=0.05,
            diffuse=0.18,
            specular=0.08,
            shadow=False,
        )

        blended = (
            key * 0.72 +
            fill * 0.20 +
            rim * 0.08
        )

        samples.append(blended.astype(np.float32))

    img = np.mean(samples, axis=0)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # -------------------------------------------------
    # SSAA DOWNSAMPLE
    # -------------------------------------------------

    pil = PILImage.fromarray(img)

    pil = pil.resize(
        (self.width, self.height),
        PILImage.LANCZOS,
    )

    return np.array(pil)
```

---

# 4. ЗАМЕНИ `_render_pass()`

```python
def _render_pass(
    self,
    W,
    H,
    view,
    proj,
    light_dir,
    light_color,
    ambient,
    diffuse,
    specular,
    shadow=True,
):

    _, _, rgba, _, _ = p.getCameraImage(
        width=W,
        height=H,
        viewMatrix=view,
        projectionMatrix=proj,

        renderer=p.ER_TINY_RENDERER,

        lightDirection=light_dir,
        lightColor=light_color,
        lightDistance=3.0,

        shadow=1 if shadow else 0,

        lightAmbientCoeff=ambient,
        lightDiffuseCoeff=diffuse,
        lightSpecularCoeff=specular,
    )

    arr = np.array(rgba, dtype=np.float32)[:, :, :3]

    # broken renderer protection
    if arr.mean() < 1:
        raise RuntimeError("Renderer returned black frame")

    return arr
```

---

# 5. YCB BLACK OBJECT FIX

В `_place_ycb()`:

Замени:

```python
p.changeVisualShape(bid, link_idx, rgbaColor=fallback)
```

на:

```python
p.changeVisualShape(
    bid,
    link_idx,
    rgbaColor=fallback,
    specularColor=[0.08, 0.08, 0.08],
    textureUniqueId=-1,
)
```

---

# 6. BUILD RETRY PROTECTION

Полностью замени `build()`:

```python
def build(
    self,
    object_specs,
    camera_yaw=45.0,
    camera_pitch=-35.0,
    camera_distance=1.0,
):

    MAX_RETRIES = 5

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:
            self.reset()

            self._load_environment()

            objects = self._place_objects(object_specs)

            if len(objects) == 0:
                raise RuntimeError("No objects placed")

            self._settle_physics(steps=200)

            self._refresh_positions(objects)

            rgb = self._render(
                camera_yaw,
                camera_pitch,
                camera_distance,
            )

            if rgb.mean() < 3:
                raise RuntimeError("Black frame")

            if not np.isfinite(rgb).all():
                raise RuntimeError("NaN frame")

            rgb = postprocess(rgb)

            return rgb, objects

        except Exception as e:

            last_error = e

            print(
                f"[WARN] Scene build failed "
                f"(attempt {attempt+1}/{MAX_RETRIES}): {e}"
            )

    raise RuntimeError(
        f"Scene generation failed after retries: {last_error}"
    )
```

---

# 7. FLOOR / WALL FIX

## FLOOR

Замени:

```python
rgbaColor=[0.12, 0.12, 0.14, 1.0]
```

на:

```python
rgbaColor=[0.22, 0.22, 0.24, 1.0]
```

---

## WALL

Замени:

```python
rgbaColor=[0.18, 0.18, 0.20, 1.0]
```

на:

```python
rgbaColor=[0.42, 0.42, 0.45, 1.0]
```

---

# 8. RECOMMENDED SETTINGS

```python
SSAA = 2
```

---

# 9. DOCKER COMMAND

```bash
docker run --rm \
  -e DISPLAY=:99 \
  -v $(pwd)/output:/workspace/output \
  pybullet-datagen \
  bash -c "
    Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
    sleep 2
    python scripts/generate_dataset.py --scenes 10 --seed 123
  "
```

---

# 10. IMPORTANT

Для Docker + Xvfb:

НЕ используй:

```python
ER_BULLET_HARDWARE_OPENGL
```

Оставь:

```python
ER_TINY_RENDERER
```

иначе будут:

- black frames
- broken renders
- random crashes
- empty images

