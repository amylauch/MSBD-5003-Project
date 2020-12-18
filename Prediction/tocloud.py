#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import *
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
import time


# In[2]:


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

full=spark.read.csv(r'./full_final.csv', header=True, inferSchema=True)
# r = full.count()
# full = full.iloc[int(0.01*r),:]


# In[3]:


full=full.withColumn("item_count",col("item_count").cast(StringType()))
full=full.withColumn("grand_total",col("grand_total").cast(StringType()))
full=full.withColumn("payment_mode",col("payment_mode").cast(StringType()))
full=full.withColumn("promo_code",col("promo_code").cast(StringType()))
full=full.withColumn("vendor_discount_amount",col("vendor_discount_amount").cast(StringType()))
full=full.withColumn("promo_code_discount_percentage",col("promo_code_discount_percentage").cast(StringType()))
full=full.withColumn("is_favorite",col("is_favorite").cast(StringType()))
full=full.withColumn("is_rated",col("is_rated").cast(StringType()))
full=full.withColumn("driver_rating",col("driver_rating").cast(StringType()))
full=full.withColumn("deliverydistance",col("deliverydistance").cast(StringType()))
full=full.withColumn("preparationtime",col("preparationtime").cast(StringType()))
full=full.withColumn("delivery_charge",col("delivery_charge").cast(StringType()))
full=full.withColumn("serving_distance",col("serving_distance").cast(StringType()))
full=full.withColumn("prepration_time",col("prepration_time").cast(StringType()))
full=full.withColumn("discount_percentage",col("discount_percentage").cast(StringType()))
full=full.withColumn("Free Delivery",col("Free Delivery").cast(StringType()))




# In[5]:


trainingData, testData = full.randomSplit([0.8, 0.2])


# In[6]:


trainingData.cache()


# In[7]:

start_time = time.time()


tokenizer = Tokenizer(inputCol="gender"and"location_type"and"item_count"and"grand_total"and"payment_mode"and"promo_code"
                               and"vendor_discount_amount"and"promo_code_discount_percentage", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(trainingData)

predictionsDf = model.transform(testData)
predictionsDf.show()


# In[8]:

print("--- Train and predict: %s seconds ---" % (time.time() - start_time))

numSuccesses = predictionsDf.where('label == prediction').count()
numInspections = predictionsDf.count()

print ("There were %d inspections and there were %d successful predictions" % (numInspections, numSuccesses))
print("This is a %d%% success rate" % (float(numSuccesses) / float(numInspections) * 100))


# In[9]:


print(numSuccesses/numInspections)


# In[ ]:





# In[ ]:




