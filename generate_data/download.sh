# CelebA images
if ! [ -x "$(command -v kaggle)" ]; then
  echo "Make sure you have kaggle installed (pip install kaggle) and kaggle API properly located at $HOME/.kaggle/kaggle.json."
  exit 1
fi

kaggle datasets download -d jessicali9530/celeba-dataset
ZIP_FILE=celeba-dataset.zip
FOLDER=./data/CelebA
mkdir -p $FOLDER
unzip $ZIP_FILE -d $FOLDER
rm $ZIP_FILE

mv $FOLDER/list_eval_partition.csv $FOLDER/train_val_test.txt #Train/Val/Test Partition
mv $FOLDER/list_attr_celeba.csv $FOLDER/list_attr_celeba.txt #List Attributes

# CelebA aligned
unzip $FOLDER/img_align_celeba.zip -d $FOLDER
rm $FOLDER/img_align_celeba.zip
rm $FOLDER/*.csv

