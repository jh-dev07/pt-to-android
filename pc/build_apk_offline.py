"""Offline APK builder. Requires Python 3.8+, JDK 17 and prepared Android SDK.
No third party Python packages, Gradle, Maven, network, NDK or Android Studio needed.
ANDROID_HOME, JAVA_HOME must be set; SDK platform 35 and build tools 35.0.0 required.
The generated signing key is for local evaluation only. Keep it for local updates.
"""
from pathlib import Path
import os, subprocess, zipfile, shutil, sys
r=Path(__file__).resolve().parents[1]
sdk=Path(os.environ['ANDROID_HOME']); jdk=Path(os.environ['JAVA_HOME'])
b=r/'local-build';b.mkdir(exist_ok=True)
win=sys.platform=='win32'
def jtool(name):return str(jdk/'bin'/(name+('.exe' if win else '')))
def st(name):return str(sdk/'build-tools/35.0.0'/(name+('.bat' if win and name in ['d8','apksigner'] else '.exe' if win else '')))
def run(args):
    # Windows batch entry points need the command processor; paths are quoted by list2cmdline.
    if win and str(args[0]).endswith('.bat'):subprocess.run(subprocess.list2cmdline(args),shell=True,check=True)
    else:subprocess.run(args,check=True)
jar=sdk/'platforms/android-35/android.jar'
with zipfile.ZipFile(r/'android/app/libs/onnxruntime-android-1.20.0.aar') as z:z.extractall(b/'ort')
for name in ['classes','dex']:
    shutil.rmtree(b/name,ignore_errors=True);(b/name).mkdir()
src=r/'android/app/src/main/java/com/example/offlineshapes/MainActivity.java'
run([jtool('javac'),'--release','8','-classpath',os.pathsep.join([str(jar),str(b/'ort/classes.jar')]),'-d',str(b/'classes'),str(src)])
run([jtool('jar'),'cf',str(b/'app.jar'),'-C',str(b/'classes'),'.'])
run([st('d8'),'--lib',str(jar),'--min-api','23','--output',str(b/'dex'),str(b/'app.jar'),str(b/'ort/classes.jar')])
s=(r/'android/app/src/main/AndroidManifest.xml').read_text().replace('<manifest ','<manifest package="com.example.offlineshapes" ',1)
(b/'AndroidManifest.xml').write_text(s)
run([st('aapt2'),'link','-I',str(jar),'--manifest',str(b/'AndroidManifest.xml'),'--min-sdk-version','23','--target-sdk-version','35','--version-code','1','--version-name','1.0','-A',str(r/'android/app/src/main/assets'),'-o',str(b/'base.apk')])
with zipfile.ZipFile(b/'base.apk','a',zipfile.ZIP_DEFLATED) as z:
    for p in (b/'dex').glob('*.dex'):z.write(p,p.name)
    for abi in ['arm64-v8a','x86_64']:
        for p in (b/'ort/jni'/abi).glob('*.so'):z.write(p,'lib/'+abi+'/'+p.name)
run([st('zipalign'),'-f','4',str(b/'base.apk'),str(b/'aligned.apk')])
key=b/'debug.keystore'
if not key.exists():run([jtool('keytool'),'-genkeypair','-keystore',str(key),'-storepass','android','-keypass','android','-alias','androiddebugkey','-dname','CN=Android Debug,O=Android,C=US','-keyalg','RSA','-keysize','2048','-validity','10000'])
out=r/'Offline_Shapes-debug.apk'
run([st('apksigner'),'sign','--ks',str(key),'--ks-pass','pass:android','--out',str(out),str(b/'aligned.apk')])
run([st('apksigner'),'verify','--verbose',str(out)])
print('Created',out)
