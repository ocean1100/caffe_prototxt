//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;  //image w | h size
const int kCIFARImageNBytes = 3072; //image bytes number
const int kCIFARBatchSize = 10000;  //batch size
const int kCIFARTrainBatches = 5;   //train batches

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type) {
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label;
  char str_buffer[kCIFARImageNBytes];
  Datum datum;
  datum.set_channels(3);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  LOG(INFO) << "Writing Training data";
  //n batches
  
  int fileid;
  for (fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    string batchFileName = input_folder + "/data_batch_"
      + caffe::format_int(fileid+1) + ".bin";
    std::ifstream data_file(batchFileName.c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;

    if(fileid < kCIFARTrainBatches - 1){
        //n images per batch
        for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
          read_image(&data_file, &label, str_buffer);
          datum.set_label(label);
          datum.set_data(str_buffer, kCIFARImageNBytes);
          string out;
          CHECK(datum.SerializeToString(&out));
          txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
        }
    }
    else{
        //pick 5000 of the last batch as train, the other 5000 as validation
        for (int itemid = 0; itemid < kCIFARBatchSize/2; ++itemid) {
            read_image(&data_file, &label, str_buffer);
            datum.set_label(label);
            datum.set_data(str_buffer, kCIFARImageNBytes);
            string out;
            CHECK(datum.SerializeToString(&out));
            txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
        }
/***************************************************************************/
        txn->Commit();
        train_db->Close();
        LOG(INFO) << "Writing Validating data";
        scoped_ptr<db::DB> validate_db(db::GetDB(db_type));
        validate_db->Open(output_folder + "/cifar10_validate_" + db_type, db::NEW);
        txn.reset(validate_db->NewTransaction());
        // Open files
        //std::ifstream data_file((input_folder + "/data_batch_" + caffe::format_int(fileid+1) + ".bin").c_str(),
        //    std::ios::in | std::ios::binary);
        //CHECK(data_file) << "Unable to open validate file.";
        //data_file.seek
        for (int itemid = 0; itemid < kCIFARBatchSize/2; ++itemid) {
          read_image(&data_file, &label, str_buffer);
          datum.set_label(label);
          datum.set_data(str_buffer, kCIFARImageNBytes);
          string out;
          CHECK(datum.SerializeToString(&out));
          txn->Put(caffe::format_int(itemid, 5), out);
        }
        txn->Commit();
        validate_db->Close();
/**************************************************************************/
    }
  }


/**************************************************************************/
  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, kCIFARImageNBytes);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(caffe::format_int(itemid, 5), out);
  }
  txn->Commit();
  test_db->Close();

}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;

  if (argc != 4) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
