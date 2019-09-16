config 1 : generator loss : 0.01 * image_loss + 0.1 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss (model saved as new_gen_epoch)
config 2 : generator loss : 0.01 * image_loss + 1 * adversial_loss + 0.006 * perception_loss + 2e-8 * tv_loss (model saved as New3_gen_epoch)


The directory that is used in this code contains three folders 'test', 'train' and 'degraded'

the test folder contained all the test images 
the train folder contained all the train images
the degraded folder contained all the test images with degraded function applied. (this folder is used for testing) the test folder is just used for extracting the original image for comparison.
