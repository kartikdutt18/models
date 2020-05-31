/**
 * @file augmentation_impl.hpp
 * @author Kartik Dutt
 * 
 * Implementation of Augmentation class for augmenting data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
// Incase it has not been included already.
#include "augmentation.hpp"

#ifndef MODELS_AUGMENTATION_IMPL_HPP
#define MODELS_AUGMENTATION_IMPL_HPP

Augmentation::Augmentation() :
    augmentations(std::vector<std::string>()),
    augmentationProbability(0.2)
{
  // Nothing to do here.
}
#endif