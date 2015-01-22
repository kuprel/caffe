// This program converts a set of image pairs to a leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_videoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
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

#define NUM_FRAMES 3

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

DEFINE_bool(gray, false, "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false, "Randomly shuffle the order of images and their labels");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

bool ReadVideoToDatum(const string& root_folder, string filenames[NUM_FRAMES], 
  const int label, const int height, const int width, const bool is_color, Datum* datum) {
  
  string filepaths[NUM_FRAMES];
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  vector<cv::Mat> cv_imgs, cv_img_origins;
  int num_channels = 3;

  for(int i=0; i<NUM_FRAMES; ++i) {
    filepaths[i] = root_folder + filenames[i];
    cv_img_origins.push_back(cv::imread(filepaths[i], cv_read_flag));
    cv_imgs.push_back(cv_img_origins[i]);
    if (height > 0 && width > 0) {
      cv::resize(cv_img_origins[i], cv_imgs[i], cv::Size(width, height));
    }
  }

  datum->set_channels(num_channels*NUM_FRAMES);
  datum->set_height(cv_imgs[0].rows);
  datum->set_width(cv_imgs[0].cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();

  for(int i=0; i<NUM_FRAMES; ++i) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_imgs[i].rows; ++h) {
        for (int w = 0; w < cv_imgs[i].cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_imgs[i].at<cv::Vec3b>(h, w)[c]));
        }
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
  ifstream infile(argv[2]);
  typedef pair<string[NUM_FRAMES], int> Line;
  Line line;
  vector<Line> lines;
  string l;
  while(getline(infile, l)) {
    istringstream iss(l);
    for (int i=0; i<NUM_FRAMES; i++) iss >> line.first[i];
    iss >> line.second;
    lines.push_back(line);
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const char* db_path = argv[3];

  int resize_height = max<int>(0, FLAGS_resize_height);
  int resize_width = max<int>(0, FLAGS_resize_width);

  // Open new db
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;
  LOG(INFO) << "Opening leveldb " << db_path;
  leveldb::Status status = leveldb::DB::Open(options, db_path, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_path << ". Does it already exist?";
  batch = new leveldb::WriteBatch();

  // Store to db
  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;

  for (int i=0; i<lines.size(); ++i) {
    if (!ReadVideoToDatum(root_folder, lines[i].first,
        lines[i].second, resize_height, resize_width, is_color, &datum)) {
      continue;
    }
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size " << data.size();
    }
    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d", i);
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
