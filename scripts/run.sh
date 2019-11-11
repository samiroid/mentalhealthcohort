PROJECT_PATH="/Users/samir/Dev/projects/mental_health_cohort/"
# PROJECT_PATH="/data/ASMAT/ASMAT/experiments/user_models/"
DATA=$PROJECT_PATH"/DATA/processed"
RESFILE="results"
RESULTS=$DATA"/results/"$RESFILE
FEATURES=$DATA"/pkl/features"
MODELS=$DATA"/models"
DATASET=$1
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
TWEETS="mental_health_tweets"
# rm experiments/demos/DATA/txt/mental_health_*

#extract mental health data
# python src/get_mentalhealth_data.py

#
#extract vocabulary and indices	
#NOTE embedding based models can represent all words in the embedding matrix so it is 
# ok to include the test set in the vocabulary

python src/users_extract.py -labels_path $DATA"/txt/"$TRAIN \
                                            $DATA"/txt/"$DEV $DATA"/txt/"$TEST \
                                            -text_path $DATA"/txt/"$TWEETS \
                                -vocab_from $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
                                -vocab_size 100000 \
                                -out_folder $FEATURES 

python src/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
							-out_folder $FEATURES \
							-bow bin freq							

python src/linear_model.py -features bow-bin \
										-train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
							 			-res_path $RESULTS
							 
	python src/linear_model.py -features bow-freq	 \
										-train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
										-res_path $RESULTS 
										