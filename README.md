# Diploma
Diploma project for hand gesture CDT (Classification Detection Tracking)

## Todo pipeline

1. Clone or download https://github.com/gustavz/deeptraining_hands
 Here you can find pretty scripts for easy dataset download and preparation. The only class will be called "hand"
 Here run:\
    oxfordhands_setup.py\
    egohands_setup.py
    
2. To create csv format annotations do not forget to run\
    xml_to_csv.py\
    To generate record check your data/labels.pbtxt is correct and run\
    csv_to_tfrecord.py
 
3. If you have lots of classes and want easily to create label-map when having xml-annotations run the following modified script from https://github.com/douglasrizzo/detection_util_scripts :\
    create_label_map.py\
 And then generate the tf.record