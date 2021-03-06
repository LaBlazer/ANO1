// DIP.cpp : Defines the entry point for the console application.
//
#pragma once
#include "stdafx.h"
#include "kmeans.h"
#include "backprop.h"

#define p(m, x, y) m.at<uchar>(y, x)

enum Shape {
	Square,
	Rectangle,
	Star
};

template<typename T>
struct Vec2 {
	T x, y;
};

struct Object {
	int idx;
	Vec2<double> center;
	double f1, f2;
	Shape shape;
};

struct ObjectClass {
	Vec2<double> ethalon;
	Shape shape;
};

void flood(int32_t x, int32_t y, cv::Mat& in_mat, cv::Mat& out_mat, uchar idx) {
	const auto size = in_mat.size();
	if (x < 0 || x >= size.width || y < 0 || y >= size.height)
		return;

	if (p(out_mat, x, y) > 0)
		return;

	//printf("%d %d\n", x, y);

	p(out_mat, x, y) = idx;

	if (p(in_mat, x + 1, y) > 0)
		flood(x + 1, y, in_mat, out_mat, idx);
	
	if (p(in_mat, x, y + 1) > 0)
		flood(x, y + 1, in_mat, out_mat, idx);

	if (p(in_mat, x - 1, y) > 0)
		flood(x - 1, y, in_mat, out_mat, idx);

	if (p(in_mat, x, y - 1) > 0)
		flood(x, y - 1, in_mat, out_mat, idx);


	if (p(in_mat, x + 1, y - 1) > 0)
		flood(x + 1, y - 1, in_mat, out_mat, idx);

	if (p(in_mat, x + 1, y + 1) > 0)
		flood(x + 1, y + 1, in_mat, out_mat, idx);

	if (p(in_mat, x - 1, y + 1) > 0)
		flood(x - 1, y + 1, in_mat, out_mat, idx);

	if (p(in_mat, x - 1, y - 1) > 0)
		flood(x - 1, y - 1, in_mat, out_mat, idx);
}

int coordinate_moment(cv::Mat& mat, int p, int q, uchar idx) {
	int m = 0;
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			if (p(mat, x, y) == idx) {
				m += pow(x, p) * pow(y, q);
			}
		}
	}
	return m;
}

int center_moment(cv::Mat& mat, int p, int q, Vec2<double>& center, uchar idx) {
	int m = 0;
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			if (p(mat, x, y) == idx) {
				m += pow(x - center.x, p) * pow(y - center.y, q);
			}
		}
	}
	return m;
}

Vec2<double> object_center(cv::Mat& mat, uchar index) {
	const int m00 = coordinate_moment(mat, 0, 0, index);

	Vec2<double> p;
	p.x = coordinate_moment(mat, 1, 0, index) / (double)m00;
	p.y = coordinate_moment(mat, 0, 1, index) / (double)m00;

	return p;
}

int object_circumference(cv::Mat& mat, int idx) {
	int circumference = 0;
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			if (p(mat, x, y) == idx) {
				if (p(mat, x + 1, y) != idx ||
					p(mat, x, y + 1) != idx ||
					p(mat, x - 1, y) != idx ||
					p(mat, x, y - 1) != idx) {
					circumference += 1;
				}
			}
		}
	}
	return circumference;
}

Vec2<double> calculate_ethalon(Shape shape, std::vector<Object> objects) {
	Vec2<double> out{0, 0};
	int count = 0;
	for (auto& o : objects) {
		if (o.shape == shape) {
			out.x += o.f1;
			out.y += o.f2;
			count++;
		}
	}

	out.x /= count;
	out.y /= count;
	return out;
}

std::tuple<double, double> u_minmax(cv::Mat& mat, int idx, Vec2<double>& center) {
	double first_part = 0.5 * (center_moment(mat, 2, 0, center, idx) + center_moment(mat, 0, 2, center, idx));
	double second_part = 0.5 * sqrt(4 * pow(center_moment(mat, 1, 1, center, idx), 2) + pow(center_moment(mat, 2, 0, center, idx) - center_moment(mat, 0, 2, center, idx), 2));
	return std::make_tuple<double, double>(first_part - second_part, first_part + second_part);
}

std::string shape_to_string(Shape s) {
	switch (s) {
	case Shape::Rectangle:
		return "Rectangle";
	case Shape::Square:
		return "Square";
	case Shape::Star:
		return "Star";
	}
	return "";
}

std::vector<Object> detect_objects(cv::Mat input_img, const int treshold = 128) {

	auto size = input_img.size();

	cv::Mat input_img_treshold(size.height, size.width, CV_8UC1);
	cv::Mat input_img_flood(size.height, size.width, CV_8UC1);
	input_img_flood.setTo(0);

	for (int x = 0; x < size.width; x++) {
		for (int y = 0; y < size.height; y++) {
			if (input_img.at<uchar>(y, x) > treshold) {
				input_img_treshold.at<uchar>(y, x) = 255;
			}
			else {
				input_img_treshold.at<uchar>(y, x) = 0;
			}
		}
	}

	std::cout << "Detecting objects.... " << std::endl;

	std::vector<Object> objects;
	uchar flood_idx = 1;
	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			auto col = input_img_treshold.at<uchar>(y, x) > 0;
			auto flooded = input_img_flood.at<uchar>(y, x) > 0;
			if (col && !flooded) {
				flood(x, y, input_img_treshold, input_img_flood, flood_idx);

				Object o;
				o.idx = flood_idx;
				objects.emplace_back(std::move(o));

				flood_idx++;
			}
		}
	}

	for (auto& o : objects) {
		o.center = object_center(input_img_flood, o.idx);
		o.f1 = pow(object_circumference(input_img_flood, o.idx), 2) / (100. * center_moment(input_img_flood, 0, 0, o.center, o.idx));

		const auto minmax = u_minmax(input_img_flood, o.idx, o.center);
		o.f2 = std::get<0>(minmax) / std::get<1>(minmax);

		std::cout << "Object " << o.idx << " - F1: " << o.f1 << " F2: " << o.f2 << std::endl;
	}

	return objects;
}

double euclidean_distance(double x1, double y1, double x2, double y2)
{
	const double dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
	return sqrt(dist);
}


Shape detect_shape(ObjectClass classes[3], Object object) {
	double min_distance = 99999999;
	Shape shape;
	for (int i = 0; i < 3; i++) {
		const double dist = euclidean_distance(classes[i].ethalon.x, classes[i].ethalon.y, object.f1, object.f2);
		if (dist < min_distance) {
			min_distance = dist;
			shape = classes[i].shape;
		}
	}

	return shape;
}

void manual_train() {
	cv::Mat train_img = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (train_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	auto objects = detect_objects(train_img);

	// AI matrix quantum blockchain recurrent neural network training
	objects[0].shape = Shape::Square;
	objects[1].shape = Shape::Square;
	objects[2].shape = Shape::Square;
	objects[3].shape = Shape::Square;
	objects[4].shape = Shape::Star;
	objects[5].shape = Shape::Star;
	objects[6].shape = Shape::Star;
	objects[7].shape = Shape::Star;
	objects[8].shape = Shape::Rectangle;
	objects[9].shape = Shape::Rectangle;
	objects[10].shape = Shape::Rectangle;
	objects[11].shape = Shape::Rectangle;

	ObjectClass classes[3];
	classes[0].ethalon = calculate_ethalon(Shape::Square, objects);
	classes[0].shape = Shape::Square;
	classes[1].ethalon = calculate_ethalon(Shape::Star, objects);
	classes[1].shape = Shape::Star;
	classes[2].ethalon = calculate_ethalon(Shape::Rectangle, objects);
	classes[2].shape = Shape::Rectangle;

	
	// load test objects
	cv::Mat test_img = cv::imread("images/test01.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (test_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	auto test = detect_objects(test_img);

	// detect shapes
	for (auto& o : test) {
		const auto shape = detect_shape(classes, o);
		cv::putText(test_img, shape_to_string(shape), cv::Point(o.center.x, o.center.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(128, 128, 128), 2);
		std::cout << "Object " << o.idx << " shape: " << shape_to_string(shape) << std::endl;
	}

	// diplay images
	cv::imshow("Training image", train_img);
	cv::imshow("Test image", test_img);
}

void kmeans_train() {
	cv::Mat train_img = cv::imread("images/test01.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (train_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	auto objects = detect_objects(train_img);
	auto data = std::make_shared<std::vector<std::array<double, 2>>>();

	for (auto& o : objects) {
		data->push_back({ o.f1, o.f2 });
	}

	kmeans<2> kmeans(3, data);

	auto clusters = kmeans.make_clusters();

	for (int i = 0; i < clusters.size(); i++) {
		std::cout << clusters[i].centroid[0] << ' ' << clusters[i].centroid[1] << std::endl;

		for (int oi : clusters[i].indices) {
			cv::putText(train_img,
				std::to_string(i),
				cv::Point(objects[oi].center.x, objects[oi].center.y),
				cv::FONT_HERSHEY_PLAIN, 1,
				cv::Scalar(128, 128, 128), 3);
		}
	}

	// diplay images
	cv::imshow("Kmeans image", train_img);
}

void train(NN* nn)
{	
	cv::Mat train_img = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (train_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	auto objects = detect_objects(train_img);

	// AI matrix quantum blockchain recurrent neural network training
	objects[0].shape = Shape::Square;
	objects[1].shape = Shape::Square;
	objects[2].shape = Shape::Square;
	objects[3].shape = Shape::Square;
	objects[4].shape = Shape::Star;
	objects[5].shape = Shape::Star;
	objects[6].shape = Shape::Star;
	objects[7].shape = Shape::Star;
	objects[8].shape = Shape::Rectangle;
	objects[9].shape = Shape::Rectangle;
	objects[10].shape = Shape::Rectangle;
	objects[11].shape = Shape::Rectangle;

	const int n = objects.size();
	double** trainingSet = new double* [n];
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[5];

		trainingSet[i][0] = objects[i].f1;
		trainingSet[i][1] = objects[i].f2;
		trainingSet[i][2] = (objects[i].shape == Shape::Square) ? 1.0 : 0.0;
		trainingSet[i][3] = (objects[i].shape == Shape::Star) ? 1.0 : 0.0;
		trainingSet[i][4] = (objects[i].shape == Shape::Rectangle) ? 1.0 : 0.0;
	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i % n]);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, const char* filename)
{
	cv::Mat test_img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	if (test_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}

	auto objects = detect_objects(test_img);
	double* in = new double[2];

	for (auto& o : objects) {
		in[0] = o.f1;
		in[1] = o.f2;

		setInput(nn, in, true);
		feedforward(nn);
		int output = getOutput(nn, true);

		cv::putText(test_img,
			std::to_string(output),
			cv::Point(o.center.x, o.center.y),
			cv::FONT_HERSHEY_PLAIN, 1,
			cv::Scalar(128, 128, 128), 3);
	}

	cv::imshow("BNN image", test_img);
}

void bnn_train() {
	NN* nn = createNN(2, 4, 3);

	train(nn);

	//getchar();

	test(nn, "images/test01.png");

	//getchar();

	releaseNN(nn);

	// neuralka
}

int main()
{
	//manual_train();
	//kmeans_train();
	bnn_train();

	cv::waitKey(0); // wait until keypressed

	return 0;
}
