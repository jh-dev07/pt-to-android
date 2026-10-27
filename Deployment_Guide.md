# Deploying PC Trained AI Models on Android

A practical workflow and offline example for native Java applications

Prepared 27 October 2026

## Purpose and outcome

This guide explains how to take trained model files from a PC, identify what they contain, restore the original model, convert it into a mobile runtime format, and integrate it into an Android Java application. It also explains how to verify correctness and prepare for offline operation.

The accompanying Offline Shapes example contains an actual trained PyTorch checkpoint, its exported ONNX model, an Android Java project, an Android runtime AAR, deterministic test inputs, and a standalone browser demonstration. Its small network classifies synthetic filled circles, squares, and triangles. It demonstrates deployment mechanics rather than production image recognition. The same learned weights are used across the PC, ONNX, and browser paths.

An Android model deployment normally consists of four things: the model graph and weights, a compatible runtime, matching preprocessing, and output interpretation. Copying a checkpoint into an APK does not provide the other three. Training remains on the PC; inference runs on the device.

## 1 Identify the model before choosing a converter

The filename extension is a clue, not a reliable format specification. A .pt file might contain a state dictionary, a general training checkpoint, a serialized Python module, or a TorchScript program. A .bin file might contain Hugging Face weights, another framework's tensors, a quantized model, or unrelated binary data. Renaming a file changes none of its contents.

Collect the model architecture or original repository, framework versions, checkpoint creation code, license, input specification, output specification, and required sidecar files. For a Hugging Face model this may include config.json, tokenizer.json, tokenizer configuration, special token maps, vocabulary files, processor configuration, and every weight shard named by an index. For OCR, speech, and detection models also collect alphabet files, feature extraction settings, label maps, and decoding parameters.

If you have only weights and no architecture, conversion usually cannot proceed reliably. Weights give tensor values; they do not necessarily describe the computation that connects those tensors. Ask the training author for the inference code and an example input with expected output.

Inspect the code that saved the checkpoint. torch.save(model.state_dict(), path) saves parameters and buffers. torch.save({'model_state_dict': ..., 'optimizer_state_dict': ...}, path) creates a broader training checkpoint. torch.save(model, path) serializes a Python object and may depend on the original class import path. torch.jit.save saves a TorchScript artifact. These cases need different loaders.

Load only trusted artifacts. Use torch.load(..., map_location='cpu', weights_only=True) for compatible weight dictionaries. Do not switch to unrestricted pickle loading merely to suppress an error. A trusted legacy full-module checkpoint may require its original environment and an explicit migration into a state dictionary.

## 2 Select the Android runtime

ExecuTorch is a PyTorch deployment path that exports supported models into .pte programs. LiteRT provides a .tflite route for supported models, including PyTorch conversion through LiteRT Torch. ONNX Runtime provides Android inference for supported .onnx graphs and optimized .ort artifacts. None is a universal converter for every model.

Use operator coverage, available export recipes, supported devices, accelerator support, runtime size, and maintenance requirements to choose. Begin with a CPU implementation that produces correct results. Add GPU or NPU acceleration after the baseline passes. Accelerators can introduce device restrictions, unsupported operators, partition overhead, precision differences, or separate export requirements.

The example uses PyTorch to ONNX Runtime because its small linear and ReLU graph has a straightforward conversion and a Java Android API. This is an illustrative choice, not a claim that ONNX is best for every model. A supported LLM export recipe, for example, may be more practical through ExecuTorch, LiteRT LM, or another model-specific runtime.

The example pins older, known versions for reproducibility: PyTorch 2.5.1 CPU, ONNX 1.17.0, PC ONNX Runtime 1.20.1, Android ONNX Runtime 1.20.0, ONNX opset 17, and ONNX IR version 8 from the PyTorch exporter. These are tested example dependencies, not a recommendation to use them unchanged for all new production applications. Review supported versions and security updates before shipping.

## 3 Establish a PC inference baseline

Restore the exact architecture, load its weights, move the model to CPU for a simple baseline, and call eval(). Evaluation mode changes behavior for modules such as dropout and batch normalization. Use torch.no_grad() or inference_mode() while running predictions. Keep training-only optimizer state outside the Android deliverable.

Create representative test cases before exporting. Preserve raw input, the final numeric tensor supplied to the model, expected outputs, and decoded results. Include ordinary inputs, empty inputs, maximum supported lengths or sizes, and difficult cases relevant to the actual application. Do not use only the training set.

Write down the input contract. For an image model record RGB versus BGR, resize and crop policy, pixel range, mean and standard deviation, channel order, batch dimension, orientation, and alpha handling. For audio record sampling rate, channel count, window size, feature extraction, and normalization. For text record tokenizer version, special tokens, truncation, padding, attention masks, and generation settings.

Record output names, tensor shapes, types, label order, and decoding. A class index has no useful meaning without the matching label list. Detector outputs require box decoding and potentially nonmaximum suppression. LLM logits require a tokenizer and a generation loop; an ordinary classifier wrapper does not implement these features.

## 4 Export and verify the converted model

Export on the PC, using representative tensors and an opset compatible with the target runtime. Start with fixed shapes when the application can use them. Dynamic dimensions make some models easier to use but can reduce export or accelerator compatibility. Model export can execute model code, so keep the source and checkpoint trusted.

The example defines ShapeNet in pc/train_export.py. It is a float32 network with 256 inputs, a 48-unit ReLU hidden layer, and three output logits. It is trained on 6000 generated examples and evaluated on 900 separately generated examples. The input shape is exactly [1, 256]; flattening uses row-major order, with index y * 16 + x. A filled grid cell is 1.0 and an empty cell is 0.0. Output order is Circle, Square, Triangle.

The training script saves shapes.pt, reconstructs a new ShapeNet, loads that checkpoint with weights_only=True, and exports that restored model into shapes.onnx. This exercises the checkpoint restoration step instead of exporting only the in-memory training object. The script also saves validation tensors, browser weights, and three fixture predictions from the restored PyTorch model.

Run onnx.checker.check_model, then compare the restored PyTorch outputs against ONNX Runtime outputs using the same input tensors. Compare numerical outputs as well as decoded predictions. Use tolerances appropriate to the model and precision; exact byte equality is usually inappropriate for floating point computation. The bundled float32 validation uses rtol=1e-4 and atol=1e-4. Its Android fixture test permits an absolute logit error of 0.001.

If export fails, inspect the first unsupported operator, data-dependent Python control flow, custom extension, or dynamic shape error. Simplify the inference wrapper or replace unsupported operations deliberately. A lower opset is not a guaranteed fix. If the file exports but predictions change, inspect preprocessing and output decoding before attributing everything to the converter.

## 5 Integrate the model into native Android Java

Add the matching Android runtime to the application and bundle the model under app/src/main/assets. This example ships a local onnxruntime-android-1.20.0.aar in app/libs, avoiding a Maven fetch for the inference library itself. Android Gradle Plugin and other build tools still require preparation separately.

MainActivity creates one OrtEnvironment and one reusable OrtSession. It reads the small model from assets and creates a float32 OnnxTensor with shape [1, 256]. session.run receives a map whose key is the exported input name pixels. The output is a float array containing three logits. The application uses the largest logit to select a label.

Inference and session initialization run on a single background executor. This keeps model work off the UI thread and serializes session use. The input grid is copied before it is queued, so later touch events do not change an input already submitted for inference. Output and input tensors are closed using try-with-resources. The session is closed during activity teardown after pending work drains.

The manifest declares no INTERNET permission. The app loads no URLs and includes no model download logic. It needs no Python interpreter, PC server, login, or cloud account. Inference uses CPU execution; GPU and NPU use are intentionally outside the example's tested configuration.

The model is small enough for an asset byte array. Large production models may require installation into app-private storage, memory mapping, external tensor data, or a different packaging scheme. Do not assume the small-model loading code is appropriate for multi-gigabyte models.

## 6 Try the offline example

Extract Android_Model_Offline_Example.zip. For the fastest test, open Offline_Shapes.html in a desktop browser. It has all weights, fixtures, scripts, and styles embedded. It does not fetch a model or load a CDN. The page runs its fixture self-test at startup. Click Circle, Square, or Triangle, then Classify. Clear the grid and paint a centered filled shape to explore the model. The browser executes the learned matrix operations in JavaScript; it is a reference implementation, not an Android runtime benchmark.

For Android, install the supplied debug APK, launch Offline Shapes, and tap Run offline self test. Expected output is PASS for all three fixture predictions with maximum logit error below 0.001. Tap each sample button to classify the matching filled shape. Drawings far outside the synthetic training distribution can receive confidently wrong predictions. The displayed softmax value is a model score, not calibrated probability of correctness.

Turn airplane mode on before launching, disable Wi-Fi if airplane mode leaves it enabled, force-stop the app, and launch again. Repeat the fixture test and sample predictions. This verifies a cold offline launch, rather than only an already loaded session. Repeat on arm64-v8a physical hardware and an x86_64 emulator if both are target environments. This APK configuration does not include 32-bit ABIs.

With Android platform-tools available, install using adb install -r Offline_Shapes-debug.apk. If several devices are connected, run adb devices and use adb -s DEVICE_SERIAL install -r Offline_Shapes-debug.apk. For LDPlayer, use the adb executable supplied by the emulator when appropriate. Installation permission and emulator Android version still apply; the example minimum Android version is API 23, Android 6.0.

## 7 Build and retrain when needed

The Android source uses Gradle 8.9, Android Gradle Plugin 8.7.3, JDK 17, compile SDK 35, and build tools selected by that plugin. It uses native platform widgets and has no AppCompat dependency. These versions belong to this example; do not mix them with a previous project's Gradle configuration without checking compatibility.

On a prepared connected development machine, open the android folder in Android Studio or use the Gradle wrapper. Set JAVA_HOME to JDK 17 and ANDROID_HOME or sdk.dir to the installed SDK. Run ./gradlew :app:assembleDebug on Linux, or gradlew.bat :app:assembleDebug on Windows. The APK is generated under app/build/outputs/apk/debug/app-debug.apk. A debug APK is for evaluation; a production release requires your own signing and release process.

To retrain on a PC, create a virtual environment and install the dependencies in pc/requirements.txt. The tested environment is CPython 3.12 on Linux x86_64; choose platform-matching wheels on Windows or other hosts. Install a CPU PyTorch build where appropriate instead of unnecessary CUDA packages. Run python pc/train_export.py from the package root, then python pc/validate.py. No external dataset is used. Rebuilding the Android APK is necessary after changing its bundled assets.

The shipped browser page embeds the original weights. Retraining alone does not regenerate that page. Use pc/make_browser.py after training to refresh the page and its JavaScript parity test, then rebuild and revalidate the deliverables. Existing installed APKs also retain their original model until updated or given a separately implemented model-update mechanism.

To make a new isolated PC ready for offline conversion, download all Python wheels on a connected machine with the same operating system, architecture, and Python version. For example, python -m pip download -r pc/requirements.txt -d wheelhouse. Select the correct PyTorch index and wheel deliberately. Transfer the complete wheelhouse and install with python -m pip install --no-index --find-links wheelhouse -r pc/requirements.txt. The package does not include a cross-platform Python wheelhouse. Opening the browser demo or running the APK requires none of these Python packages.

For this example, an alternative direct SDK builder is included in pc/build_apk_offline.py. With Python 3.8 or newer, JDK 17, SDK platform android-35 and build tools 35.0.0 already installed, set JAVA_HOME and ANDROID_HOME and run python pc/build_apk_offline.py. It compiles the Java classes, runs D8, packages the model and native libraries, aligns the APK, creates a local evaluation key, and signs and verifies the result. It needs no Gradle, Maven, NDK, external Python packages, or network access. This path was tested on Linux; the Windows branch is supplied but not executed here. It supports this resource-free example rather than arbitrary Android projects.

The supplied evaluation APK and your own rebuilt APK may use different signing keys. Android rejects an update signed by a different key. Retain your local key for subsequent local builds, and use your own protected release key for production. Native operation on devices using 16 KB memory pages has not been validated with the pinned runtime.

## 8 Prepare an offline Android build machine

Offline runtime and offline compilation are separate requirements. The supplied model and AAR satisfy only part of the build dependency graph. A new disconnected PC also needs the JDK, Android SDK platform, build tools, Gradle distribution, Android Gradle Plugin, and all transitive plugin dependencies.

Prepare the same project on a connected machine using the same operating system and architecture as the offline build host. Resolve a full debug build, then disconnect and run ./gradlew --offline --no-daemon :app:assembleDebug --rerun-tasks. The wrapper must also have its Gradle distribution cached. A successful connected build alone does not prove the cache is complete for offline rebuilding.

Transfer the project, JDK, relevant SDK components, Gradle distribution, and the complete required Gradle cache or an explicitly prepared local Maven repository. Respect vendor redistribution licenses and accept required SDK licenses on the target host. Avoid hardcoded user-specific SDK paths and do not commit local.properties. Set paths on the destination machine.

On Windows, Gradle caches usually live under %USERPROFILE%\.gradle and the SDK often under %LOCALAPPDATA%\Android\Sdk. On Linux, the corresponding Gradle cache is usually ~/.gradle. Cache reuse is sensitive to tool versions and native artifacts. A cache copied from Linux is not evidence that Windows builds will work offline.

On the offline target run the same --offline build with networking disconnected. If Gradle reports a missing artifact, prepare that artifact on the connected machine and repeat the test. The flag prevents repository access; it does not download missing dependencies. The delivered source archive is not a complete offline Android Studio installer or SDK distribution.

## 9 Optimize and validate on real devices

First measure model initialization, preprocessing, inference, postprocessing, and total interaction time separately. Use warm-up runs before steady-state timing, then record median and p95 latency over many runs. Measure cold starts separately. Report device model, Android version, ABI, backend, thread count, and power or thermal conditions.

Quantization can reduce model size and memory use, but it can also change accuracy and operator support. Evaluate FP16, dynamic INT8, static INT8 with representative calibration data, or model-specific low-bit recipes where supported. A four-bit LLM recipe is not interchangeable with generic integer quantization for a vision model. Keep the original float model as a reference.

Estimate runtime memory beyond the model file: parameters, intermediate tensors, workspace, preprocessing buffers, and application memory all contribute. LLMs also require attention cache memory that increases with context and batch size. A small compressed weight file does not guarantee low peak RAM.

Add accelerators only after checking backend support and testing the actual target devices. Android versions, chipsets, drivers, and ABI packaging affect availability. Include a tested CPU fallback if your product requires it. Do not infer Android latency from a desktop benchmark.

For correctness, validate representative held-out data, numerical parity, decoded outputs, blank and malformed inputs, and maximum-size inputs. For reliability, test screen rotation, background and foreground transitions, repeated requests, low-memory conditions, and repeated cold launches. Run offline with networking disabled and inspect the manifest for unexpected network permissions introduced by dependencies.

## 10 Replace the demo with your own model

Keep the original architecture and trusted checkpoint together. Write an export wrapper that exposes only the inference inputs and outputs. Restore the model in evaluation mode and verify a PC prediction. Export to the chosen mobile runtime and compare a representative test set. Document the final tensor contract alongside the exported model.

Update the Java input preparation, tensor shapes, input names, output parsing, and label mapping together. Merely replacing shapes.onnx with a different model will generally fail because MainActivity expects a float32 tensor called pixels with shape [1, 256] and three logits. A camera classifier needs image preprocessing; speech requires audio handling; a language model needs tokenization and generation.

Use model version metadata and a checksum to identify each deployed artifact. For independently delivered model updates, download or transfer into a temporary location, verify integrity and compatibility, create and smoke-test a new session, then activate it atomically. Retain the previous compatible model for rollback. In a fully disconnected environment, the update artifact can arrive by USB or managed local transfer.

A model update can alter learned behavior without an APK update when the installed app already supports that model contract and file-loading path. New native operations, unsupported tensor contracts, Java logic, or runtime upgrades usually require an application update. Protect signing keys and retain your original application signing identity for future installations over the existing app.

## Troubleshooting

File will not load: confirm that it is a supported exported runtime model rather than a training checkpoint. Check format, runtime version, graph operators, opset, and whether external tensor files are missing.

Unexpected output: compare the exact tensor supplied on PC and Android. Check channel order, normalization, flattening, shape, label order, tokenizer files, and output decoding before retraining.

Unsupported operator: identify the operation and target backend. Use a supported export recipe, adjust the model wrapper, or choose another runtime. Custom operators require implementation and packaging on Android.

Native library load error: inspect the APK for the device ABI and confirm compatible minimum Android version. This example includes arm64-v8a and x86_64 only.

Offline build fails: identify the exact missing Gradle, SDK, or Python artifact. Prepare it while connected for the same target platform. A local inference AAR does not include the Android compiler toolchain.

Out of memory or slow response: profile loading and inference separately, reuse sessions, reduce unnecessary input size and copies, consider a smaller architecture or quantization, and benchmark on the actual device.

Incorrect hand drawing: the demonstration was trained on synthetic filled upright shapes with limited size and translation variation. It has no unknown class and does not establish real handwriting recognition quality. Use the bundled sample fixtures to test deployment correctness.

## Official references

PyTorch model saving and loading: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

PyTorch ONNX export: https://docs.pytorch.org/docs/stable/onnx.html

ONNX Runtime Java API introduction: https://onnxruntime.ai/docs/get-started/with-java.html

ONNX Runtime mobile deployment: https://onnxruntime.ai/docs/tutorials/mobile/

ONNX Runtime Android installation: https://onnxruntime.ai/docs/install/

ExecuTorch getting started: https://docs.pytorch.org/executorch/stable/getting-started-setup.html

LiteRT PyTorch converter: https://github.com/google-ai-edge/litert-torch

Gradle dependency caching and offline mode: https://docs.gradle.org/current/userguide/dependency_caching.html

Android Gradle Plugin 8.7 compatibility: https://developer.android.com/build/releases/past-releases/agp-8-7-0-release-notes

## Validation record

PC parity validation passed on 900 synthetic holdout inputs. Browser fixture parity passed. TEST_RESULTS.json records measured results and limitations. The APK passed build and signature checks; Android device execution was not performed. Run its built-in fixture test on your device.
