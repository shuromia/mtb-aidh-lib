#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

typedef struct {
    uint32_t          Frame1_M;
    uint32_t          Frame1_I;
    uint32_t          Frame1_O;
    uint32_t          Frame2_M;
    uint32_t          Frame2_I;
    uint32_t          Frame2_O;
    uint32_t          Frame3_M;
    uint32_t          Frame3_I;
    uint32_t          Frame3_O;
} CapsenseValues_t;


/**
 * \brief Set up the cap_val classification using TensorFlow Lite Micro
 *
 * \return Returns values as specified by \ref TfLiteStatus
*/
int CapClassificationSetup(void);


/**
 * \brief       Run inference on the classification model
 * \details     This prepares the values read by the Capsense for use with the model and runs
 *              inference on the model. It returns an index of the status LUT (look-up table)
 *              which has the highest probability and is therefore the best prediction.
 *
 * \param[in]   cap_val       Struct containing cap values read from the Capsense
 * \param[out]  index         Pointer for writing the identified status index to
 * \param[out]  probability   Pointer for writing the probability of the identified color to
 *
 * \return Returns values as specified by \ref TfLiteStatus
*/
int CapClassificationPerformInference(CapsenseValues_t* cap_val, int8_t* cap_index, uint8_t* probability);


#ifdef __cplusplus
}  /* extern "C" */
#endif  /* __cplusplus */
