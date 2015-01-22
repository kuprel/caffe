// This program converts a set of image pairs to a leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   file1 file2 label
//   ....

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "leveldb", "The backend for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

bool ReadImagePairToDatum(const string& root_folder, const std::pair<string, string>& image_pair, 
  const int label, const int height, const int width, const bool is_color, Datum* datum) {
  
  string filename1 = root_folder + image_pair.first;
  string filename2 = root_folder + image_pair.second;

  cv::Mat cv_img1, cv_img2;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin1 = cv::imread(filename1, cv_read_flag);
  cv::Mat cv_img_origin2 = cv::imread(filename2, cv_read_flag);
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin1, cv_img1, cv::Size(width, height));
    cv::resize(cv_img_origin2, cv_img2, cv::Size(width, height));
  } else {
    cv_img1 = cv_img_origin1;
    cv_img2 = cv_img_origin2;
  }

  int num_channels = 3;
  datum->set_channels(num_channels*2);
  datum->set_height(cv_img1.rows);
  datum->set_width(cv_img1.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < num_channels; ++c) {
    for (int h = 0; h < cv_img1.rows; ++h) {
      for (int w = 0; w < cv_img1.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img1.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  for (int c = 0; c < num_channels; ++c) {
    for (int h = 0; h < cv_img2.rows; ++h) {
      for (int w = 0; w < cv_img2.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img2.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_image_pairs");
    return 1;
  }

  bool is_color = !FLAGS_gray;
  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::pair<string, string>, int> > lines;
  string filename1, filename2;
  int label;
  while (infile >> filename1 >> filename2 >> label) {
    lines.push_back(std::make_pair(std::make_pair(filename1, filename2), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const string& db_backend = FLAGS_backend;
  const char* db_path = argv[3];

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path
        << ". Is it already existing?";
    batch = new leveldb::WriteBatch();
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Storing to db
  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    if (!ReadImagePairToDatum(root_folder, lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color, &datum)) {
      continue;
    }
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
          << data.size();
    }
    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.first.c_str());
    string value;
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    batch->Put(keystr, value);

    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      batch = new leveldb::WriteBatch();
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
