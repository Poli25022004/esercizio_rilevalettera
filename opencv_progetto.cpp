#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

int main()
{

    cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/saved_model.pb");
    if (net.empty()) {
        std::cout << "Errore caricamento modello\n";
        return -1;
    }

    std::vector<std::string> labels;
    std::ifstream file("model/labels.txt");
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Errore apertura webcam\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray, blurImg, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
        cv::Canny(blurImg, edges, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (auto& c : contours) {
            cv::Rect rect = cv::boundingRect(c);

            if (rect.width < 20 || rect.height < 20) continue;
            if (rect.width > 200 || rect.height > 200) continue;

            cv::Mat roi = frame(rect);
            cv::Mat roiResized;
            cv::resize(roi, roiResized, cv::Size(224, 224));
            cv::cvtColor(roiResized, roiResized, cv::COLOR_BGR2RGB);
            cv::Mat blob = cv::dnn::blobFromImage(roiResized, 1.0 / 255.0, cv::Size(224, 224));

            net.setInput(blob);
            cv::Mat output = net.forward();

            cv::Point classId;
            double confidence;
            cv::minMaxLoc(output, 0, &confidence, 0, &classId);
            int idx = classId.x;

            std::string text;
            if (confidence < 0.6 || labels[idx] == "tastiera") {
                text = "Nessuna lettera";
            }
            else {
                text = labels[idx];
            }

            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, text, cv::Point(rect.x, rect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Riconoscimento tasti", frame);
        if (cv::waitKey(1) == 27) break; 
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
