/**
 * @file object_classification.hpp
 * @author Kartik Dutt
 *
 * Contains implementation of object classification suite. It can be used
 * to select object classification model, it's parameter dataset and
 * other training parameters.
 *
 * NOTE: This code needs to be adapted as this implementation doesn't support
 *       Command Line Arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <dataloader/dataloader.hpp>
#include <models/models.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

int main()
{
  arma::mat input, labels, output;
  DataLoader<> dataloader;
  dataloader.LoadAllImagesFromDirectory("./../../../../imagenette_new/n01440764/", input, labels, 224, 224, 3, 448);
  PreProcessor<>::ChannelFirstImages(input, 224, 224, 3);
  FFN<mlpack::ann::CrossEntropyError<>> model;
  mlpack::data::Load("../darknet19_imagenet.bin", "DarkNet", model);
  mlpack::data::Load("./../../../../imagenette_image.csv", input);
  if (input.n_cols > 80)
  {
      input = input.t();
      cout << "New cols : " << input.n_cols << std::endl;
  }
  for (size_t i = 0; i < input.n_cols; i++)
  {
      output.clear();
      cout << i << " : " << "  " << arma::accu(input.col(i)) << " ";
      model.Predict(input.col(i), output);
      cout << output.index_max() << std::endl;
      input(arma::span(0, 10), arma::span(i)).print();
  }

  return 0;
}
