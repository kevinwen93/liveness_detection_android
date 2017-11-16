package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;
import org.w3c.dom.Text;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.SeekBar.OnSeekBarChangeListener;

public class FdActivity extends Activity implements CvCameraViewListener2 {

	private static final String TAG = "OCVSample::Activity";
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
	public static final int JAVA_DETECTOR = 0;
	private static final int TM_SQDIFF = 0;
	private static final int TM_SQDIFF_NORMED = 1;
	private static final int TM_CCOEFF = 2;
	private static final int TM_CCOEFF_NORMED = 3;
	private static final int TM_CCORR = 4;
	private static final int TM_CCORR_NORMED = 5;


	private int learn_frames = 0;
	private Mat teplateR;
	private Mat teplateL;
	private Mat reflectR;
	private Mat reflectL;
	int method = 0;

	private MenuItem mItemFace50;
	private MenuItem mItemFace40;
	private MenuItem mItemFace30;
	private MenuItem mItemFace20;
	private MenuItem mItemType;

	private Mat mRgba;
	private Mat mGray;
	// matrix for zooming
//	private Mat mZoomWindow;
//	private Mat mZoomWindow2;

	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;
	private CascadeClassifier mJavaDetectorEyeR;
	private CascadeClassifier mJavaDetectorEyeL;


	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSize = 0;

	private CameraBridgeViewBase mOpenCvCameraView;
	private double[] eyeR,eyeL,refR,refL=new double[2];

	//record eyes pupil position here
	private double[] bbr, bbl, bwr, bwl, wbr, wbl, wwr, wwl;
	private int ronscreen = 0;
	private double[] dffl,dffr;
	private int countr, countl;

	double xCenter = -1;
	double yCenter = -1;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
				case LoaderCallbackInterface.SUCCESS: {
					Log.i(TAG, "OpenCV loaded successfully");


					try {
						// load cascade file from application resources
						InputStream is = getResources().openRawResource(
								R.raw.lbpcascade_frontalface_improved);
						File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
						mCascadeFile = new File(cascadeDir,
								"lbpcascade_frontalface_improved.xml");
						FileOutputStream os = new FileOutputStream(mCascadeFile);

						byte[] buffer = new byte[4096];
						int bytesRead;
						while ((bytesRead = is.read(buffer)) != -1) {
							os.write(buffer, 0, bytesRead);
						}
						is.close();
						os.close();

						// --------------------------------- load left eye
						// classificator -----------------------------------
						InputStream iser = getResources().openRawResource(
								R.raw.ojor);
						File cascadeDirER = getDir("cascadeER",
								Context.MODE_PRIVATE);
						File cascadeFileER = new File(cascadeDirER,
								"ojor.xml");
						FileOutputStream oser = new FileOutputStream(cascadeFileER);


						/////////////////////////////////////////////////////
						InputStream iserL=getResources().openRawResource(
								R.raw.ojol);
						File cascadeDirERL = getDir("cascadeERL",
								Context.MODE_PRIVATE);
						File cascadeFileERL = new File(cascadeDirER,
								"ojol.xml");
						FileOutputStream oserL = new FileOutputStream(cascadeFileERL);

						byte[] bufferER = new byte[4096];
						int bytesReadER;
						while ((bytesReadER = iser.read(bufferER)) != -1) {
							oser.write(bufferER, 0, bytesReadER);
						}
						while ((bytesReadER = iserL.read(bufferER)) != -1) {
							oserL.write(bufferER, 0, bytesReadER);
						}
						iser.close();
						iserL.close();
						oser.close();
						oserL.close();

						mJavaDetector = new CascadeClassifier(
								mCascadeFile.getAbsolutePath());
						if (mJavaDetector.empty()) {
							Log.e(TAG, "Failed to load cascade classifier");
							mJavaDetector = null;
						} else
							Log.i(TAG, "Loaded cascade classifier from "
									+ mCascadeFile.getAbsolutePath());


						////////////////////////////////////////////////////
						mJavaDetectorEyeR = new CascadeClassifier(
								cascadeFileER.getAbsolutePath());
						if (mJavaDetectorEyeR.empty()) {
							Log.e(TAG, "Failed to load cascade classifier");
							mJavaDetectorEyeR = null;
						} else
							Log.i(TAG, "Loaded cascade classifier from "
									+ mCascadeFile.getAbsolutePath());
						cascadeDirER.delete();


						////////////////////////////////////////////////
						mJavaDetectorEyeL = new CascadeClassifier(
								cascadeFileERL.getAbsolutePath());
						if (mJavaDetectorEyeL.empty()) {
							Log.e(TAG, "Failed to load cascade classifier");
							mJavaDetectorEyeL = null;
						} else
							Log.i(TAG, "Loaded cascade classifier from "
									+ mCascadeFile.getAbsolutePath());
						cascadeDirERL.delete();





					} catch (IOException e) {
						e.printStackTrace();
						Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
					}
					mOpenCvCameraView.setCameraIndex(1);
					mOpenCvCameraView.enableFpsMeter();
					mOpenCvCameraView.enableView();

				}
				break;
				default: {
					super.onManagerConnected(status);
				}
				break;
			}
		}
	};

	public FdActivity() {
		mDetectorName = new String[2];
		mDetectorName[JAVA_DETECTOR] = "Java";

		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);



		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);




	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
				mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
//		mZoomWindow.release();
//		mZoomWindow2.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		dffl = new double[2];
		dffr = new double[2];

		if (mAbsoluteFaceSize == 0) {
			int height = mGray.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
			}
		}

//		if (mZoomWindow == null || mZoomWindow2 == null)
//	        CreateAuxiliaryMats();

		MatOfRect faces = new MatOfRect();

		if (mJavaDetector != null)
			mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2,
					2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
					new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
					new Size());

		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++) {
			Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
					FACE_RECT_COLOR, 3);
			xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
			yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
			Point center = new Point(xCenter, yCenter);

			Core.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

//			Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
//					new Point(center.x + 20, center.y + 20),
//					Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
//							255));

			Rect r = facesArray[i];
			// compute the eye area
//			Rect eyearea = new Rect(r.x + r.width / 8,
//					(int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8,
//					(int) (r.height / 3.0));
			// split it
			Rect eyearea_right = new Rect(r.x,
					(int) (r.y + (r.height / 10)),
					r.width/2, (int) (r.height / 4.5));
			Rect eyearea_left = new Rect(r.x +r.width/2,
					(int) (r.y + (r.height / 10)),
					r.width/2, (int) (r.height / 4.5));
			// draw the area - mGray is working grayscale mat, if you want to
			// see area in rgb preview, change mGray to mRgba
			Core.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
					new Scalar(255, 0, 0, 255), 2);
			Core.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
					new Scalar(255, 0, 0, 255), 2);

			if (learn_frames < 3) {
				teplateR =get_template(mJavaDetectorEyeR, eyearea_right, 30, 0);
				teplateL = get_template(mJavaDetectorEyeL, eyearea_left, 30, 0);
				reflectR= get_template(mJavaDetectorEyeR, eyearea_right, 30, 1);
				reflectL= get_template(mJavaDetectorEyeL, eyearea_left, 30, 1);
				learn_frames++;
			} else {
				// Learning finished, use the new templates for template
				// matching
				if (!teplateR.empty()) {
				/*	Core.putText(mRgba, "right pupil found",
							new Point(center.x - 200, center.y + 20),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
					eyeR=match_eye(eyearea_right, teplateR, 2, 0);
					runOnUiThread(new Runnable() {

						public void run() {
							TextView eyeTR=(TextView)findViewById(R.id.eyeRText);
							eyeTR.setText("Right pupil position:\n[" +Double.toString(eyeR[0]) + "," + Double.toString(eyeR[1]) + "]");
						}
					});
				}
				else{
					/*Core.putText(mRgba, "right pupil not found",
							new Point(center.x - 200, center.y + 20),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
				}

				///////////////////////////////////////////////////////////
				if (!teplateL.empty()) {
				/*	Core.putText(mRgba, "left pupil found",
							new Point(center.x + 10, center.y + 20),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
					eyeL=match_eye(eyearea_left, teplateL, 2, 0);
					runOnUiThread(new Runnable() {

						public void run() {
							TextView eyeTL=(TextView)findViewById(R.id.eyeLText);
							eyeTL.setText("Left pupil position:\n[" +Double.toString(eyeL[0]) + "," + Double.toString(eyeL[1]) + "]");
						}
					});

				}
				else{
					/*Core.putText(mRgba, "left pupil not found",
							new Point(center.x + 10, center.y + 20),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
				}

				//////////////////////////////////////////////////////////
				if (!reflectR.empty()) {
					/*Core.putText(mRgba, "right reflection found",
							new Point(center.x - 230, center.y + 50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
					refR=match_eye(eyearea_right, reflectR, 2, 1);
					runOnUiThread(new Runnable() {

						public void run() {
							TextView refTR=(TextView)findViewById(R.id.refRText);
							refTR.setText("Right reflection position:\n[" +Double.toString(refR[0]) + "," + Double.toString(refR[1]) + "]");
						}
					});
				}
				else{
					/*Core.putText(mRgba, "right reflection not found",
							new Point(center.x - 230, center.y + 50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
				}

				///////////////////////////////////////////////////////
				if (!reflectL.empty()) {
					/*Core.putText(mRgba, "left reflection found",
							new Point(center.x + 20, center.y + 50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
					refL=match_eye(eyearea_left, reflectL, 2, 1);
					runOnUiThread(new Runnable() {

						public void run() {
							TextView refTL=(TextView)findViewById(R.id.refLText);
							refTL.setText("Left reflection position:\n[" +Double.toString(refL[0]) + "," + Double.toString(refL[1]) + "]");
						}
					});

				}
				else{
					/*Core.putText(mRgba, "left reflection not found",
							new Point(center.x + 20, center.y + 50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));*/
				}

				ImageView black=(ImageView)findViewById(R.id.blackV);
				ImageView white=(ImageView)findViewById(R.id.whiteV);
				if ((black.getVisibility()==View.VISIBLE || white.getVisibility()==View.VISIBLE) &&learn_frames==3){
					runOnUiThread(new Runnable() {
						final double[] eyeRB=eyeR;
						final double[] eyeLB=eyeL;
						final double[] refRB=refR;
						final double[] refLB=refL;
						public void run() {

							TextView refTRB=(TextView)findViewById(R.id.refRB);
							TextView refTLB=(TextView)findViewById(R.id.refLB);
							TextView eyeTRB=(TextView)findViewById(R.id.eyeRB);
							TextView eyeTLB=(TextView)findViewById(R.id.eyeLB);
							refTRB.setText("[" +Double.toString(refRB[0]) + "," + Double.toString(refRB[1]) + "]");
							refTLB.setText("[" +Double.toString(refLB[0]) + "," + Double.toString(refLB[1]) + "]");
							eyeTRB.setText("[" +Double.toString(eyeRB[0]) + "," + Double.toString(eyeRB[1]) + "]");
							eyeTLB.setText("[" +Double.toString(eyeLB[0]) + "," + Double.toString(eyeLB[1]) + "]");

						}

					});

					if(black.getVisibility() == View.VISIBLE){
						bbr = eyeR;
						bbl = eyeL;
						bwr = refR;
						bwl = refL;
					}

					if(white.getVisibility()==View.VISIBLE){
						wbr = eyeR;
						wbl = eyeL;
						wwr = refR;
						wwl = refL;
					}

				}
				learn_frames++;


				if(ronscreen == 1){

					dffl[0] = Math.abs(wbl[0]-bbl[0]);
					//dffl[1] = Math.abs(wbl[1]-bbl[1]);
					dffl[1] = Math.abs(wwl[0]-bwl[0]);
					//dffl[3] = Math.abs(wwl[1]-bwl[1]);

					dffr[0] = Math.abs(wbr[0]-bbr[0]);
					//dffl[1] = Math.abs(wbl[1]-bbl[1]);
					dffr[1] = Math.abs(wwr[0]-bwr[0]);
					//dffl[3] = Math.abs(wwl[1]-bwl[1]);

					Core.putText(mRgba, "["+Double.toString(wbl[0]-bbl[0])+","+Double.toString(wbl[1]-bbl[1])+"]",
							new Point(center.x-200, center.y+20 ),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
					Core.putText(mRgba, "["+Double.toString(wbr[0]-bbr[0])+","+Double.toString(wbr[1]-bbr[1])+"]",
							new Point(center.x+10, center.y+20),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
					Core.putText(mRgba, "["+Double.toString(wwl[0]-bwl[0])+","+Double.toString(wwl[1]-bwl[1])+"]",
							new Point(center.x-200, center.y+50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));
					Core.putText(mRgba, "["+Double.toString(wwr[0]-bwr[0])+","+Double.toString(wwr[1]-bwr[1])+"]",
							new Point(center.x+10, center.y+50),
							Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));


				}

			}
			countl = 0;
			countr = 0;

			if(dffl[0] > 10)
				countl++;
			if(dffr[0] > 10)
				countr++;
			if(dffl[1] > 17)
				countl+=2;
			if(dffr[1] > 17)
				countr+=2;

			if(countl+countr>=4){
				Core.putText(mRgba, "You are a real person!",
						new Point(center.x-10, center.y+90),
						Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));
			}else{
				Core.putText(mRgba, "Not a real person!",
						new Point(center.x-10, center.y+90),
						Core.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 255, 255, 255));
			}

			// cut eye areas and put them to zoom windows
//			Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2,
//					mZoomWindow2.size());
//			Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow,
//					mZoomWindow.size());


		}

		return mRgba;
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemFace50 = menu.add("Face size 50%");
		mItemFace40 = menu.add("Face size 40%");
		mItemFace30 = menu.add("Face size 30%");
		mItemFace20 = menu.add("Face size 20%");
		mItemType = menu.add(mDetectorName[mDetectorType]);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item == mItemFace50)
			setMinFaceSize(0.5f);
		else if (item == mItemFace40)
			setMinFaceSize(0.4f);
		else if (item == mItemFace30)
			setMinFaceSize(0.3f);
		else if (item == mItemFace20)
			setMinFaceSize(0.2f);
		else if (item == mItemType) {
			int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
			item.setTitle(mDetectorName[tmpDetectorType]);
		}
		return true;
	}

	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}


	private void CreateAuxiliaryMats() {
		if (mGray.empty())
			return;

		int rows = mGray.rows();
		int cols = mGray.cols();

//		if (mZoomWindow == null) {
//			mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
//					+ cols / 10, cols);
//			mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
//					+ cols / 10, cols);
//		}

	}

	private double[] match_eye(Rect area, Mat mTemplate, int type, int choice) {
		double[] location=new double[2];
		Point matchLoc;
		Mat mROI = mGray.submat(area);
		int result_cols = mROI.cols() - mTemplate.cols() + 1;
		int result_rows = mROI.rows() - mTemplate.rows() + 1;
		// Check for bad template size
		if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
			return location;
		}
		Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);
		Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);

		Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
		// there is difference in matching methods - best match is max/min value
		if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
			matchLoc = mmres.minLoc;
		} else {
			matchLoc = mmres.maxLoc;
		}
//		matchLoc=mmres.maxLoc;

		Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
		Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
				matchLoc.y + mTemplate.rows() + area.y);

		if (choice==0){
			Core.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0,
					255));
		}
		else{
			Core.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 0, 255,
					255));
		}

		Core.putText(mRgba, "[" + matchLoc.x + "," + matchLoc.y + "]",
				new Point(matchLoc_tx.x + 20, matchLoc_ty.y + 20),
				Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
						255));
		Rect rec = new Rect(matchLoc_tx,matchLoc_ty);
		location[0]=matchLoc.x;
		location[1]=matchLoc.y;
		return location;


	}

	private Mat get_template(CascadeClassifier clasificator, Rect area, int size, int choice) {
		// choice 0 for matching the minimum
		// choice 1 for matching the maximum
		Mat template = new Mat();
		Mat mROI = mGray.submat(area);
		MatOfRect eyes = new MatOfRect();
		Point iris = new Point();
		Point reflection=new Point();
		Rect eye_template = new Rect();
		Rect reflection_template=new Rect();
		clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
				Objdetect.CASCADE_FIND_BIGGEST_OBJECT
						| Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
				new Size());

		Rect[] eyesArray = eyes.toArray();
		for (int i = 0; i < eyesArray.length;) {
			Rect e = eyesArray[i];
			e.x = area.x + e.x;
			e.y = area.y + e.y;
			Rect eye_only_rectangle = new Rect((int) e.tl().x,
					(int) (e.tl().y + e.height * 0.4), (int) e.width,
					(int) (e.height * 0.6));
			mROI = mGray.submat(eye_only_rectangle);
			Mat vyrez = mRgba.submat(eye_only_rectangle);


			// Core.rectangle(mRgba,eye_only_rectangle.tl(),eye_only_rectangle.br(), new Scalar(134,43,23));


			Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);
			if (choice==0){
				Core.circle(vyrez, mmG.minLoc, 2, new Scalar(0, 255, 0, 255), 2);
				iris.x = mmG.minLoc.x + eye_only_rectangle.x;
				iris.y = mmG.minLoc.y + eye_only_rectangle.y;
				eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
						- size / 2, size, size);
				Core.rectangle(mRgba, eye_template.tl(), eye_template.br(),
						new Scalar(255, 0, 0, 255), 2);
				Core.putText(mRgba, "[" + iris.x + "," + iris.y + "]",
						new Point(iris.x + 20, iris.y + 20),
						Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
								255));
				template = (mGray.submat(eye_template)).clone();

			}
			else{
				Core.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 0, 255, 255), 2);
				reflection.x= mmG.maxLoc.x+eye_only_rectangle.x;
				reflection.y= mmG.maxLoc.y + eye_only_rectangle.y;
				reflection_template = new Rect((int) reflection.x - size / 2, (int) reflection.y
						- size / 2, size, size);
				Core.rectangle(mRgba,reflection_template.tl(),reflection_template.br(),
						new Scalar(0, 0, 255, 255), 2);
				Core.putText(mRgba, "[" + reflection.x + "," + reflection.y + "]",
						new Point(reflection.x + 20, reflection.y + 20),
						Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
								255));
				template =  (mGray.submat(reflection_template)).clone();
			}

			return template;
		}
		return template;
	}

	public void onRecreateClick(View v)
	{
		learn_frames = 0;
	}

	public void onMatch(View v) throws InterruptedException {
		Random r=new Random();
		ronscreen = 0;
		double sec=r.nextDouble()*5*1000;
		final long s=(long)sec;
		Handler mHandler = new Handler();
		mHandler.postDelayed(new Runnable() {
			@Override
			public void run() {
				((ImageView)findViewById(R.id.blackV)).setVisibility(View.VISIBLE);
				learn_frames=0;

			}

		}, 3000);

		mHandler.postDelayed(new Runnable() {

			@Override
			public void run() {
				((ImageView)findViewById(R.id.blackV)).setVisibility(View.GONE);
			}

		}, 6000);


		mHandler.postDelayed(new Runnable() {
			@Override
			public void run() {
				((ImageView)findViewById(R.id.whiteV)).setVisibility(View.VISIBLE);
				learn_frames = 0;
			}
		},6000);

		mHandler.postDelayed(new Runnable() {

			@Override
			public void run() {
				((ImageView)findViewById(R.id.whiteV)).setVisibility(View.GONE);
				ronscreen = 1;
			}

		}, 9000);

	}






}
