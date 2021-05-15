from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import datetime
import json
import numpy as np
import sys


def expandVisits(date_range_start, visits_by_day):
    start = datetime.datetime(*map(int, date_range_start[:10].split('-')))
    return [[(start + datetime.timedelta(days=i)).year,
             (start + datetime.timedelta(days=i)).strftime("%m-%d"), visit_per_day]
            for i, visit_per_day in enumerate(json.loads(visits_by_day))]



def main(sc, spark):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''

    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]

    CAT_CODES = {'445210', '722515', '445299', '445120', '452210', '311811',
                 '722410', '722511', '445220', '445292', '445110', '445291',
                 '445230', '446191', '446110', '722513', '452311'}

    CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3,
                 '722513': 4, '446191': 5, '446110': 5, '722515': 6, '311811': 6,
                 '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7,
                 '445210': 7, '445110': 8}

    dfD = dfPlaces \
        .filter((F.col('naics_code').isin(CAT_CODES))) \
        .select(F.col('placekey'), F.col('naics_code')) \
        .distinct()

    udfToGroup = F.udf(lambda x: CAT_GROUP[x])

    dfE = dfD.withColumn('group', udfToGroup('naics_code'))

    dfF = dfE.drop('naics_code').cache()

    groupCount = {int(row['group']): row['count'] for row in dfF.groupBy('group').count().collect()}

    def computeStats(group, visits):
        list_size = groupCount[int(group)]
        n = len(visits)
        added_list = [0] * list_size
        extended_visits = visits + added_list
        median = int(np.median(extended_visits).astype(int))
        low = int(median - np.std(extended_visits).astype(int))
        high = int(median + np.std(extended_visits).astype(int))
        return [median, max(low, 0), max(high, 0)]

    visitType = T.StructType([T.StructField('year', T.IntegerType()),
                                T.StructField('date', T.StringType()),
                              T.StructField('visits', T.IntegerType())])

    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

    dfH = dfPattern.join(dfF, 'placekey') \
        .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
        .select('group', 'expanded.*')

    statsType = T.StructType([T.StructField('median', T.IntegerType()),
                              T.StructField('low', T.IntegerType()),
                              T.StructField('high', T.IntegerType())])

    udfComputeStats = F.udf(computeStats, statsType)

    dfI = dfH.groupBy('group', 'year', 'date') \
        .agg(F.collect_list('visits').alias('visits')) \
        .withColumn('stats', udfComputeStats('group', 'visits'))

    dfJ = dfI \
        .withColumn('date', F.concat(F.lit('2020-'), F.col('date'))) \
        .select('group', 'year', 'date', 'stats.*') \
        .sort('group', 'year', 'date') \
        .cache()

    food_store = ['big_box_grocers', 'convenience_stores', 'drinking_places',
                  'full_service_restaurants', 'limited_service_restaurants',
                  'pharmacies_and_drug_stores', 'snack_and_bakeries',
                  'specialty_food_stores', 'supermarkets_except_convenience_stores']

    for group, filename in enumerate(food_store):
        dfJ.filter(f'group={group}') \
            .drop('group') \
            .coalesce(1) \
            .write.csv(f'{OUTPUT_PREFIX}/{filename}', mode='overwrite', header=True)


if __name__ == '__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)

