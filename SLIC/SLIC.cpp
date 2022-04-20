#include "stdafx.h"

#define p(m, x, y) m.at<cv::Vec3b>(y, x)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define CLAMP(val, min, max) MAX(MIN(val, max), min)
#define SQUARE(x) (x) * (x)

inline int get_grad(const cv::Mat &grad, const int x, const int y) {
    const auto pix = p(grad, x, y);
    return pix[0] + pix[1] + pix[2];
}

inline const std::array<int, 5> get_center(const cv::Mat& img, const int x, const int y) {
    const auto pix = p(img, x, y);
    return { pix[0], pix[1], pix[2], x, y }; // Rk, Gk, Bk, xk, yk
}

inline const cv::Scalar get_color(const int val, const int max) {
    const int part = 255 / max * val;
    return cv::Scalar(part, 0, 255 - part);
}

int main()
{
    const int K = 15;
    const float m = 0.8f;
    const int move_treshold = 5;
    const char* filename = "images/bears.jpg";

    // do not edit below
    cv::Mat orig_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);

    if (orig_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

    cv::Mat display_img = orig_img.clone();
    

    // find K centers
    const int width = orig_img.cols, height = orig_img.rows;
    const int segment_size = sqrt((width * height) / K);
    std::vector<std::array<int, 5>> centers;

    std::cout << "Segment size: " << segment_size << std::endl;

    for (int y = segment_size / 2; y < height; y += segment_size) {
        for (int x = segment_size / 2; x < width; x += segment_size) {

            cv::circle(display_img, { x, y }, 4, get_color(centers.size(), K), -1);
            
            const auto center = get_center(orig_img, x, y);
            centers.emplace_back(center);


            std::cout << "Center at: X" << x << " Y" << y << " \tRGB: " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
        }
    }
    
    cv::imshow("SLIC", display_img);
    cv::waitKey(0);

    // move to lowest gradient position
    cv::Mat grad_img;
    cv::Laplacian(orig_img, grad_img, -1);

    for (auto &center : centers)
    {
        int new_x = center[3], new_y = center[4];
        int grad, lowest_grad = get_grad(grad_img, new_x, new_y);

        std::cout << "Processing center X" << new_x << " Y" << new_y << std::endl;

        for (int x = MAX(center[3] - 1, 0); x < MIN(center[3] + 1, width); x++) {
            for (int y = MAX(center[4] - 1, 0); y < MIN(center[4] + 1, height); y++) {
                grad = get_grad(grad_img, x, y);

                if (grad < lowest_grad) {
                    std::cout << "New lowest gradient " << grad << " at X" << x << " Y" << y << std::endl;

                    lowest_grad = grad;
                    new_x = x;
                    new_y = y;
                }
            }
        }

        if (new_x != center[3] || new_y != center[4]) {
            std::cout << "Moving center to X" << new_x << " Y" << new_y << std::endl;

            const auto c = get_center(orig_img, new_x, new_y);

            center = c;
        }
    }

    // For each cluster center Ck, assign the pixels from a 2S × 2S square neighborhood around the cluster center Ck according to the distance DS
    const int double_segment = segment_size * 2;
    int closest_center = 0, move_distance = 0, drgb = 0, dxy = 0;
    float ds = 0.0, lowest_distance = 999999.;
    std::array<std::array<int32_t, 6>, K> new_centers;
    cv::Mat img_clusters(height, width, CV_8UC1, cv::Scalar(0));
    cv::Vec3b pix;

    do {
        move_distance = 0;

        for (auto& c : new_centers) {
            for(int i = 0; i < c.size(); i++)
                c[i] = 0;
        }
        
        for (int ix = 0; ix < width; ix++) {
            for (int iy = 0; iy < height; iy++) {

                //std::cout << "Processing X" << ix << " Y" << iy << std::endl;

                closest_center = 0;
                lowest_distance = 999999.;
                pix = p(orig_img, ix, iy);

                for (int ci = 0; ci < centers.size(); ci++)
                {
                    const auto& center = centers[ci];

                    if ((ix - double_segment) < center[3] && center[3] < (ix + double_segment) &&
                        (iy - double_segment) < center[4] && center[4] < (iy + double_segment)) {
                        
                        drgb = sqrt(SQUARE(center[0] - pix[0]) +
                            SQUARE(center[1] - pix[1]) + SQUARE(center[2] - pix[2]));
                        dxy = sqrt(SQUARE(center[3] - ix) + SQUARE(center[4] - iy));

                        ds = drgb + ((m / (float)segment_size) * dxy);

                        if (ds < lowest_distance) {
                            lowest_distance = ds;
                            closest_center = ci;
                            //std::cout << "lowest distance to cluster " << ds << std::endl;
                        }
                    }
                }

                // assign the closest cluster
                img_clusters.at<uchar>(iy, ix) = closest_center;

                // add coords and color for averaging
                auto& center = new_centers[closest_center];
                center[0] += pix[0];
                center[1] += pix[1];
                center[2] += pix[2];
                center[3] += ix;
                center[4] += iy;
                center[5] += 1;
            }
        }

        // Compute new cluster centers as the average vectors of all the pixels belonging to the cluster.
        for (int ci = 0; ci < centers.size(); ci++)
        {
            auto& old_center = centers[ci];
            auto& new_center = new_centers[ci];
            
            // calculate the average
            for (int i = 0; i < 5; i++)
                new_center[i] /= new_center[5];

            /*std::cout << "cluster " << ci << std::endl
                << "old center: X" << centers[ci][3] << " Y" << centers[ci][4] << std::endl
                << "new center: X" << new_center[0] << " Y" << new_center[1] << std::endl;*/

            move_distance += abs(old_center[3] - new_center[3]) + abs(old_center[4] - new_center[4]);

            // for visualization
            cv::line(display_img, { old_center[3], old_center[4] }, { new_center[3], new_center[4] }, get_color(ci, centers.size()), 2, cv::LINE_AA);

            // update the center
            for (int i = 0; i < 5; i++)
                old_center[i] = new_center[i];
        }

        // display progress
        cv::imshow("SLIC", display_img);
        cv::waitKey(1);

        std::cout << "\rCenters moved " << move_distance << "px    ";
    } while (move_distance > move_treshold);

    std::cout << std::endl;

    // create clustered image
    cv::Mat img_clusters_color(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            const auto& ci = img_clusters.at<uchar>(y, x);
            auto& center = centers[ci];
            p(img_clusters_color, x, y) = cv::Vec3b(center[0], center[1], center[2]);
        }
    }

    for (int ci = 0; ci < centers.size(); ci++)
    {
        auto& center = centers[ci];
        std::cout << "Center " << ci << " ended at: X" << center[3] << " Y" << center[4] << 
            " \tRGB: " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
        cv::circle(display_img, { center[3], center[4] }, 4, get_color(ci, centers.size()), -1);
    }

    cv::imshow("SLIC clust color", img_clusters_color);

    //cv::imshow("SLIC clust", img_clusters);

    cv::imshow("SLIC", display_img);

    cv::waitKey(0); // wait until keypressed
}
