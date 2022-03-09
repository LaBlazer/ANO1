#pragma once
#include "stdafx.h"

template<int dataSize>
class kmeans {
public:


	struct cluster {
		std::array<double, dataSize> centroid;
		std::vector<int> indices;
	};

	kmeans(int k, const std::shared_ptr<std::vector<std::array<double, dataSize>>> data);
	
	std::vector<cluster> make_clusters();

private:
	std::shared_ptr<std::vector<std::array<double, dataSize>>> m_data;
	int m_k;

	int random_range(int min, int max);
	double distance(const std::array<double, dataSize> &a, const std::array<double, dataSize> &b);
};

template<int dataSize>
inline kmeans<dataSize>::kmeans(int k, const std::shared_ptr<std::vector<std::array<double, dataSize>>> data)
{
	m_k = k;
	m_data = data;
}

template<int dataSize>
inline std::vector<typename kmeans<dataSize>::cluster> kmeans<dataSize>::make_clusters()
{
	std::vector<cluster> clusters;
	const int size = m_data->size();

	// 0. Create k centroids and assign them to random points
	const int part = size / m_k;
	for (int i = 0; i < size; i += part) {
		kmeans<dataSize>::cluster c{};
		c.centroid = m_data->at(i);

		//for (int j = i; j < i + part; j++)
		//	c.indices.push_back(j);
		
		clusters.emplace_back(c);
	}

	//for (int j = part * m_k; j < size; j++)
	//	clusters.back().indices.push_back(j);
	
	bool changed = false;
	do {

		// 1. Compute Euclidean distance from each centroid to all input data points.
		// 2. Assign each input data to the closest centroid.

		for (int i = 0; i < m_k; i++)
			clusters[i].indices.clear();

		for (int i = 0; i < size; i++) {
			int closest_cluster_idx = -1;
			double closest_dist = 99999999;

			for (int j = 0; j < m_k; j++) {
				const double dist = distance(clusters[j].centroid, m_data->at(i));

				if (dist < closest_dist) {
					closest_dist = dist;
					closest_cluster_idx = j;
				}
			}

			clusters[closest_cluster_idx].indices.push_back(i);
		}

		// 3. Update position of the centroid by calculating the mean position of the assigned points.
		// 4. Repeat from 1 until centroids do not move

		changed = false;
		for (auto& c : clusters) {

			std::array<double, dataSize> oldMean = c.centroid;
			std::array<double, dataSize> mean = {0, 0};

			for (const int i : c.indices) {
				for (int j = 0; j < dataSize; j++) {
					mean[j] = mean[j] + m_data->at(i)[j];
				}
			}

			std::cout << &c << " mean: ";
			for (int j = 0; j < dataSize; j++) {
				mean[j] = mean[j] / c.indices.size();

				std::cout << mean[j] << ' ';

				// compare to old mean
				if (mean[j] != oldMean[j]) {
					changed = true;
				}


				// set centroid to new mean
				c.centroid[j] = mean[j];
			}
			std::cout << std::endl;
		}
		
	} while (changed);
	

	

	return clusters;
}

template<int dataSize>
inline int kmeans<dataSize>::random_range(int min, int max)
{
	// i know i should be using mt19937 but idc
	return min + (rand() % static_cast<int>(max - min + 1));
}

template<int dataSize>
inline double kmeans<dataSize>::distance(const std::array<double, dataSize>& a, const std::array<double, dataSize>& b)
{
	double dist = 0.;
	for (int i = 0; i < dataSize; i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(dist);
}
