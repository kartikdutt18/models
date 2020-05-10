/**
 * @file utils_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <utils/utils.hpp>
#include <dataloader/datasets.hpp>
#include <dataloader/dataloader.hpp>
#include <models/lenet/lenet.hpp>
#include <ensmallen.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
using namespace boost::unit_test;
using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ModelTests);

template <
    class ModelType,
    class OptimizerType,
    class MetricType = mlpack::metric::SquaredEuclideanDistance,
    typename InputType = arma::mat,
    typename OutputType = arma::mat>
void CheckFFNWeights(ModelType& model,
                     const std::string &datasetName,
                     const double threshold, const bool takeMean,
                     OptimizerType &optimizer)
{
  DataLoader<InputType, OutputType> dataloader(datasetName, true);
  // Train the model. Note: Callbacks such as progress bar and loss aren't
  // used in testing. Training the model for few epochs ensures that a
  // user can use the pretrained model on any other dataset.
  model.Train(dataloader.TrainX(), dataloader.TrainY(), optimizer);
  // Verify viability of model on validation datset.
  OutputType predictions;
  model.Predict(dataloader.ValidX(), predictions);
  double error = MetricType::Evaluate(predictions, dataloader.ValidY());
  if (takeMean)
  {
    error = error / predictions.n_elem;
  }

  BOOST_REQUIRE_LE(error, threshold);
}

/**
 * Simple test for Le-Net model.
 */
BOOST_AUTO_TEST_CASE(LeNetModelTest)
{
  mlpack::ann::LeNet<> lenetModel(1, 28, 28, 10, "mnist");
  // Create an optimizer object for tests.
  ens::SGD<ens::AdamUpdate> optimizer(1e-4, 16, 1000,
      1e-8, true, ens::AdamUpdate(1e-8, 0.9, 0.999));
}

BOOST_AUTO_TEST_SUITE_END();
