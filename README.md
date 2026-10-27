# Offline Shapes

A real, tiny PyTorch model exported to ONNX and integrated into native Java Android.

## Try on Android

1. Copy `Offline_Shapes-debug.apk` to an Android device or drag it into LDPlayer.
2. Install and launch **Offline Shapes**. Minimum Android 6.0, arm64-v8a or x86_64.
3. Tap **Run offline self test**. Expect PASS for 3 of 3 fixtures.
4. Tap **Circle**, **Square**, or **Triangle**. Each loads an editable sample and runs inference.
5. Turn airplane mode on, disable Wi-Fi, force-stop, relaunch, and repeat.

No internet permission, login, Python, server, or model download is needed at runtime.
The model recognizes synthetic filled shapes; it is not a general handwriting model.

With adb already installed:

```powershell
.\adb.exe devices
.\adb.exe -s YOUR_DEVICE_SERIAL install -r Offline_Shapes-debug.apk
```

The APK uses an evaluation signing key. If you rebuild with a different key, Android will not install it over the original; use a separate test installation or retain your own signing identity. Do not use the debug key for production.

## Try without installing anything

Open `Offline_Shapes.html` locally in a desktop browser. It embeds all learned weights and fixtures and runs entirely offline. The reference uses JavaScript matrix operations; Android uses the bundled ONNX model and ONNX Runtime. Click the sample buttons and self-test. No CDN, HTTP server, or Python is needed.

## Contents

- `Offline_Shapes-debug.apk`: installable evaluation app, arm64-v8a and x86_64.
- `Offline_Shapes.html`: self contained browser reference.
- `android/`: native Java Android Studio source and Gradle wrapper.
- `android/app/libs/`: bundled ONNX Runtime Android AAR 1.20.0.
- `android/app/src/main/assets/`: exported ONNX model and parity fixtures.
- `pc/shapes.pt`: actual trained PyTorch state dictionary.
- `pc/train_export.py`: reproducible synthetic training, checkpoint loading and ONNX export.
- `pc/validate.py`: PyTorch to ONNX parity and 900-case holdout validation.
- `pc/make_browser.py`: regenerates browser demo after training.
- `pc/validation.npz`: fixed held-out tensors and labels.
- `pc/build_apk_offline.py`: dependency-free Python APK builder for a prepared SDK/JDK.
- `TEST_RESULTS.json`: numerical PC validation and Android build status.
- `licenses/`: runtime license and third-party notices.

## Model contract

Input `pixels`: float32 `[1,256]`, row-major 16 by 16 grid, empty 0 and filled 1.
Output `logits`: float32 `[1,3]`, label order Circle, Square, Triangle.
Architecture: Linear 256 to 48, ReLU, Linear 48 to 3. Fixed batch size 1.
Blank input is rejected by the UI. Nonempty arbitrary input always gets one of three labels.
The displayed softmax score is not calibrated confidence.

## Build with Gradle

Requires JDK 17, Gradle 8.9 (wrapper included), SDK platform 35 and Android build tools. Initial dependency preparation needs connectivity. SDK paths belong in your environment or your own local.properties.

```powershell
cd android
.\gradlew.bat :app:assembleDebug
# Once dependencies and wrapper distribution are cached:
.\gradlew.bat --offline --no-daemon clean :app:assembleDebug
```

On Linux use `./gradlew`. Output is `app/build/outputs/apk/debug/app-debug.apk`.
The package does not include a full SDK, JDK, Gradle cache or Android Studio installer.

## Build with a prepared SDK and no Gradle

Requires Python 3.8 or newer, JDK 17, SDK platform `android-35` and SDK build tools `35.0.0` already present. No external Python packages or Maven artifacts are needed. The local AAR supplies runtime code and native libraries. This path is for this resource-free example, not a replacement build system for arbitrary Android apps.

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:ANDROID_HOME = 'C:\path\to\Android\Sdk'
python pc\build_apk_offline.py
```

The script generates `Offline_Shapes-debug.apk` and creates a local evaluation key under `local-build`. Keep that key if you want to install later local builds over the previous local build. Your key will differ from the key used for the supplied APK.

## Retrain or validate

The tested conversion host is CPython 3.12 Linux x86_64. Windows wheels must match the target Python version. Prepare a virtual environment and install `pc/requirements.txt`, choosing the official CPU torch wheel where appropriate. Those packages are not included here; they are unnecessary for the APK and browser demo.

```text
python pc/train_export.py
python pc/validate.py
python pc/make_browser.py
```

Then rebuild the APK to include updated assets. Install offline Python packages from a prepared platform-matching wheelhouse with `pip install --no-index --find-links wheelhouse -r pc/requirements.txt`.

## Validation and limitations

PC validation passed on 900 synthetic holdout cases with 100% synthetic accuracy and maximum PyTorch-to-ONNX logit difference 0.000003815. All three browser fixture comparisons passed. This does not measure general drawing recognition or real-world image accuracy.

The APK was compiled with Gradle, rebuilt using a clean Gradle --offline build, and signature-verified. The direct SDK builder also passed on Linux. Android device and emulator execution have not been performed in this environment; use the built-in offline fixture test on your device. Desktop latency in TEST_RESULTS.json is not Android latency. Windows rebuild commands are provided but were not executed on Windows.

Native runtime compatibility on devices using 16 KB memory pages has not been validated. This example pins an older runtime for reproducibility; review supported runtime versions and native page-size compatibility for production.
