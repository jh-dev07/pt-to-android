package com.example.offlineshapes;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.view.MotionEvent;
import android.view.View;
import android.widget.*;
import ai.onnxruntime.*;
import org.json.*;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.*;

public class MainActivity extends Activity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private OrtEnvironment env;
    private OrtSession session;
    private JSONArray fixtures;
    private TextView result;
    private Grid grid;
    private volatile boolean ready=false;
    private boolean eraser=false;
    private final String[] names={"Circle","Square","Triangle"};
    private byte[] asset(String path) throws IOException {
        try(InputStream in=getAssets().open(path); ByteArrayOutputStream out=new ByteArrayOutputStream()) {
            byte[] buffer=new byte[8192]; int n;
            while((n=in.read(buffer))!=-1) out.write(buffer,0,n);
            return out.toByteArray();
        }
    }
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll=new ScrollView(this); LinearLayout page=new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL); page.setPadding(28,48,28,48); scroll.addView(page); setContentView(scroll);
        TextView title=new TextView(this); title.setText("Offline Shapes"); title.setTextSize(27); page.addView(title);
        TextView help=new TextView(this); help.setText("A tiny trained neural network, running on this device. Tap a sample or paint a centered, filled shape. Synthetic demo only."); page.addView(help);
        grid=new Grid(); page.addView(grid,new LinearLayout.LayoutParams(-1,Math.round(300*getResources().getDisplayMetrics().density)));
        LinearLayout samples=new LinearLayout(this); page.addView(samples);
        for(int i=0;i<3;i++){ final int label=i; Button b=new Button(this); b.setText(names[i]); samples.addView(b,new LinearLayout.LayoutParams(0,-2,1)); b.setOnClickListener(v->{ if(!ready)return; try{grid.pixels=fixturePixels(fixtures.getJSONObject(label)); grid.invalidate(); infer();}catch(Exception e){show(e.toString());} }); }
        LinearLayout actions=new LinearLayout(this); page.addView(actions);
        Button clear=new Button(this); clear.setText("Clear"); actions.addView(clear); clear.setOnClickListener(v->{grid.pixels=new float[256];grid.invalidate();show("Choose a sample or paint a filled shape.");});
        Button erase=new Button(this); erase.setText("Eraser off"); actions.addView(erase); erase.setOnClickListener(v->{eraser=!eraser;erase.setText(eraser?"Eraser on":"Eraser off");});
        Button classify=new Button(this); classify.setText("Classify"); page.addView(classify); classify.setOnClickListener(v->infer());
        Button tests=new Button(this); tests.setText("Run offline self test"); page.addView(tests); tests.setOnClickListener(v->{if(ready)worker.execute(this::selfTest);});
        result=new TextView(this); result.setTextSize(18); result.setText("Loading bundled ONNX model…"); page.addView(result);
        worker.execute(()->{try {env=OrtEnvironment.getEnvironment();try(OrtSession.SessionOptions options=new OrtSession.SessionOptions()){ options.setIntraOpNumThreads(2);session=env.createSession(asset("shapes.onnx"),options); } fixtures=new JSONArray(new String(asset("fixtures.json"),"UTF-8"));ready=true;selfTest();}catch(Exception e){show("Load failed: "+e);} });
    }
    private void show(String text){runOnUiThread(()->{if(!isFinishing()&&!isDestroyed())result.setText(text);});}
    private float[] fixturePixels(JSONObject fixture)throws JSONException{JSONArray a=fixture.getJSONArray("pixels");float[] x=new float[256];for(int i=0;i<256;i++)x[i]=(float)a.getDouble(i);return x;}
    private float[] run(float[] x)throws OrtException{
        try(OnnxTensor tensor=OnnxTensor.createTensor(env,FloatBuffer.wrap(x),new long[]{1,256}); OrtSession.Result output=session.run(Collections.singletonMap("pixels",tensor))){return ((float[][])output.get(0).getValue())[0].clone();}
    }
    private int best(float[] logits){int b=0;for(int i=1;i<3;i++)if(logits[i]>logits[b])b=i;return b;}
    private void infer(){
        if(!ready){show("Model is still loading.");return;}
        final float[] input=grid.pixels.clone();float sum=0;for(float v:input)sum+=v;
        if(sum==0){show("Empty grid. Choose a sample or paint a filled shape.");return;}
        worker.execute(()->{try{long start=System.nanoTime();float[] logits=run(input);double ms=(System.nanoTime()-start)/1e6;int b=best(logits);double total=0;for(float v:logits)total+=Math.exp(v-logits[b]);show(String.format(Locale.US,"%s\nModel score %.1f%% · inference %.2f ms\nScores are not calibrated confidence.",names[b],100/total,ms));}catch(Exception e){show("Inference failed: "+e);} });
    }
    private void selfTest(){try{double max=0;for(int i=0;i<fixtures.length();i++){JSONObject f=fixtures.getJSONObject(i);float[] got=run(fixturePixels(f));JSONArray expected=f.getJSONArray("logits");if(best(got)!=f.getInt("label"))throw new Exception("Wrong class on fixture "+i);for(int k=0;k<3;k++){if((Float.isNaN(got[k]) || Float.isInfinite(got[k])))throw new Exception("Nonfinite output");max=Math.max(max,Math.abs(got[k]-expected.getDouble(k)));}}if(max>0.001)throw new Exception("Parity error "+max);show("PASS · 3 of 3 bundled fixtures\nPC to Android logit error: "+max+"\nNo network permission is declared.");}catch(Exception e){show("Self test FAILED: "+e);} }
    @Override public void onDestroy(){ready=false;worker.execute(()->{try{if(session!=null)session.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}
    private class Grid extends View {
        float[] pixels=new float[256]; Paint paint=new Paint();
        Grid(){super(MainActivity.this);setContentDescription("16 by 16 shape drawing grid");}
        @Override protected void onDraw(Canvas c){float side=Math.min(getWidth(),getHeight());float cell=side/16;for(int y=0;y<16;y++)for(int x=0;x<16;x++){paint.setColor(pixels[y*16+x]>.5f?0xff235c60:0xffeef3f4);c.drawRect(x*cell+1,y*cell+1,(x+1)*cell-1,(y+1)*cell-1,paint);}}
        @Override public boolean onTouchEvent(MotionEvent e){getParent().requestDisallowInterceptTouchEvent(true);float cell=Math.min(getWidth(),getHeight())/16f;int x=(int)(e.getX()/cell),y=(int)(e.getY()/cell);if(x>=0&&x<16&&y>=0&&y<16)pixels[y*16+x]=eraser?0:1;invalidate();if(e.getAction()==MotionEvent.ACTION_UP){performClick();infer();}return true;}
        @Override public boolean performClick(){super.performClick();return true;}
    }
}
