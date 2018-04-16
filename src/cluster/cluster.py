import os.path
from pyspark.sql import SparkSession
from cluster.pyspark_kmodes import EnsembleKModes
from pyspark.sql.types import *
import pyspark.sql.functions as func

# Default values for local use.
SURVEY = os.path.join(os.path.dirname(__file__), './data/survey')
CLUSTER_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), './data/cluster')
CENTROIDS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), './data/centroids')

# Setting constants
MAX_PARTITIONS = 32
N_CLUSTERS = 2
MAX_DIST_ITER = 10


SURVEY_COL_DICT = {
    "col0": "value"
}


def cluster(**kwargs):
    survey = kwargs.get('survey', SURVEY)
    cluster_output_dir = kwargs.get('cluster_output', CLUSTER_OUTPUT_DIR)
    centroids_output_dir = kwargs.get('centroids_output', CENTROIDS_OUTPUT_DIR)

    # Initialize/set up the SparkSQL session.
    spark = SparkSession \
        .builder \
        .appName('ZCluster') \
        .getOrCreate()

    # Allow for RDD lineage graph to be truncated and saved to a file system.
    spark.sparkContext.setCheckpointDir('checkpoint/')

    centroids_fields = [StructField(SURVEY_COL_DICT[key], StringType(), True) for key in SURVEY_COL_DICT.keys() if SURVEY_COL_DICT[key] not in "user_id"]
    centroids_schema = StructType(centroids_fields)

    # Get schema
    cluster_fields = [StructField(SURVEY_COL_DICT["col0"], LongType(), True)]
    cluster_fields.extend(centroids_fields)
    cluster_schema = StructType(cluster_fields)

    # Load Survey Data for Clustering
    survey_data = spark.read.csv(survey, header=False, schema=cluster_schema)
    survey_data_rdd = survey_data.drop('user_id').rdd.coalesce(MAX_PARTITIONS)

    # Initialize the Model
    method = EnsembleKModes(n_clusters=N_CLUSTERS,
                            max_dist_iter=MAX_DIST_ITER,
                            verbosity=1)

    # Fit the model
    model = method.fit(survey_data_rdd)

    # Get centroid
    centroids = model.centroids
    centroids_list = [centroid.tolist() for centroid in centroids]
    centroids_data = spark.createDataFrame(centroids_list[1:], centroids_schema)

    # Predict and attach results
    result = spark.createDataFrame([int(model.predict(rows)) for rows in survey_data_rdd.collect()], IntegerType())

    survey_data = survey_data.withColumn('row_index', func.monotonically_increasing_id())
    result = result.withColumn('row_index', func.monotonically_increasing_id())

    survey_data = survey_data.join(result, on=["row_index"]).drop("row_index")

    # Save results
    survey_data.write \
        .mode('overwrite') \
        .format('com.databricks.spark.csv') \
        .save(cluster_output_dir)

    centroids_data.write \
        .mode('overwrite') \
        .format('com.databricks.spark.csv') \
        .save(centroids_output_dir)

    # Stop the SparkSQL session.
    spark.stop()
