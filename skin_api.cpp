#include "skin_api.h"


#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

enum SKIN_ERROR
{
    SKIN_OK = 0,
    SKIN_NO_FACE = -1,
    SKIN_TOO_MANY_FACES = -2,
    SKIN_INVALID_FEATURES = -3
};

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

int face_detector(char *fname, std::vector<matrix<rgb_pixel>>& faces)
{
     frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

    matrix<rgb_pixel> img;
    load_image(img, fname);

    // Run the face detector on the image of our action heroes, and for each face extract a
    // copy that has been normalized to 150x150 pixels in size and appropriately rotated
    // and centered.
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
    }

   return faces.size();
}


int face_gen(char *fname, float * features)
{
    std::string a = std::string(fname);
    
    std::vector<matrix<rgb_pixel>> faces;
    int face_nb = face_detector(fname, faces);
 
    if (face_nb == 0)
        return SKIN_NO_FACE;
    if (face_nb > 1)     
        return SKIN_TOO_MANY_FACES;

    // We load the DNN responsible for face recognition.
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
      
    // This call asks the DNN to convert each face image in faces into a 128D vector.
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);

    float * result = new float [face_descriptors[0].nr()];
    if (face_descriptors[0].nr() != 128)
        return SKIN_INVALID_FEATURES;  

    for (long r = 0; r < face_descriptors[0].nr(); ++r)
    {
        features[r] = face_descriptors[0](r,0);
    }
    return SKIN_OK;
}

