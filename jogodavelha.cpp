#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp" 

using namespace cv;
using namespace std;


int main(int argc, const char** argv)
{

	vector <Mat> vetorIn;
	
	vetorIn.push_back(imread("../data/jogo1.png", IMREAD_COLOR));	

	int janela = 0;
	for (auto &const im : vetorIn) {
		
		Mat out;
		cvtColor(im, out, CV_BGR2GRAY);

		//Passa filtro passa alta (bordas)
		Laplacian(out, out, CV_8U);

		//Faz o fechamento
		Mat e = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(out, out, CV_MOP_CLOSE, e);

		//Filtro Galssiano (passa-baixa) - Suavizacao e remocao de ruidos
		GaussianBlur(out, out, Size(3, 3), 3);

		//Binariza
		threshold(out, out, 50, 255, THRESH_BINARY_INV);

		//Conecta os componentes com branco
		floodFill(out, Point(0, 0), Scalar(255));

		//Aplica erosão
		erode(out, out, e);

		//Conecta os componentes com branco
		floodFill(out, Point(0, 0), Scalar(0));

		//Detecta os blobs
		vector<vector<Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(out, contours, hierarchy, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		vector<Rect> rects;
		int maxSize = 0;

		//Contabiliza os blobs
		for (int i = 0; i < contours.size(); ++i)
		{
			Rect rect = boundingRect(Mat(contours[i]));
			int size = rect.size().height*rect.size().width;
			rects.push_back(rect);

			if (maxSize < size)
				maxSize = size;
		}

		//Elimina os blobs pequenos
		int total = 0;
		for (auto &const rect : rects)
		{
			int size = rect.size().height*rect.size().width;

			if (size > maxSize * 0.75)
			{

				//Cria um Mat para o blob da imagem original
				Mat blob(Mat(out, rect));

				int noZero = countNonZero(blob);
				int size = blob.size().height * blob.size().width;

				if (noZero < size * 0.95) {
					total++;
					rectangle(im, Rect(rect.x, rect.y, rect.width, rect.height), Scalar(0, 0, 255));
				}
			}
		}


		string win = "win";
		win.append(to_string(janela++));

		cout << "Total blobs: " << total << " win " << win << endl;
		

		imshow(win, im);

	}

	
	waitKey(0);
	return 0;
}


