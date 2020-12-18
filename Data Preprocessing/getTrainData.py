#!/usr/bin/env python
# coding: utf-8

# In[49]:


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


# In[50]:


cleaned_orders=spark.read.csv(r'./cleaned_orders.csv', header=True, inferSchema=True)
cleaned_train_customer_after=spark.read.csv(r'./cleaned_train_customer.csv', header=True, inferSchema=True)
cleaned_train_locations=spark.read.csv(r'./cleaned_train_locations.csv', header=True, inferSchema=True)
cleaned_vendors=spark.read.csv(r'./cleaned_vendors.csv', header=True, inferSchema=True)
target=spark.read.csv(r'./target.csv', header=True, inferSchema=True)
full=cleaned_orders.join(cleaned_train_locations,'customer_id','outer')
cleaned_train_customer_after=cleaned_train_customer_after.withColumnRenamed('akeed_customer_id', 'customer_id')


# In[51]:


full=full.join(cleaned_train_customer_after,'customer_id','outer')
cleaned_vendors=cleaned_vendors.withColumnRenamed('authentication_id', 'akeed_order_id')
full=full.join(cleaned_vendors,'akeed_order_id','outer')
full=full.join(target,'CID X LOC_NUM X VENDOR','outer')


# In[56]:


# full=full.drop('akeed_order_id','customer_id','_c0','order_accepted_time','driver_accepted_time',
#                'ready_for_pickup_time','picked_up_time','delivered_time','delivery_date','vendor_id',
#                'created_at','_c0', 'location_number','location_type', 'latitude', 'longitude', '_c0',
#               'gender','dob', 'status_customer','verified_customer', 'language_customer', '_c0',
#                'id', 'latitude_vendor','longitude_vendor', 'vendor_category_en', 'vendor_category_id',
#               'is_open','status_vendor','verified_vendor','rank','language', 'vendor_rating', 'primary_tags',
#               'device_type','delivery_time','_c0','Fatayers','Cafe','Kebabs','Family Meal','Spanish Latte',
#               'Kushari','Frozen yoghurt','Shuwa','Churros','Bagels','Combos','Pastas','Hot Chocolate',
#               'Rolls','Karak','Smoothies','Organic','Coffee','Thai','Chinese','Seafood','Pastry','Vegetarian',
#               'Thali','Biryani','Mishkak','Pizza','Steaks','Sweets','Rice','Dimsum','Hot Dogs','Waffles',
#               'Pancakes','Ice creams','Fresh Juices','Donuts','Kids meal','Manakeesh','Omani','Mandazi','Sushi',
#               'Japanese','Healthy Food','Asian','Milkshakes','Fries','Indian','Soups','Pizzas','Pasta','Italian',
#               'Crepes','Cakes','Shawarma','Sandwiches','Salads','Lebanese','Grills','Desserts','Burgers',
#               'Breakfast','Arabic')
#full=full.drop('Mojitos','American','Mexican')
full=full.drop('akeed_order_id','customer_id','_c0','order_accepted_time','driver_accepted_time',
               'ready_for_pickup_time','picked_up_time','delivered_time','delivery_date','vendor_id',
               'created_at','_c0', '_c0','_c0','id', 'vendor_category_en', 'vendor_category_id','_c0','vendor_rating',
               'location_type', 'location_number')


# In[57]:


full=full.na.fill(0)


# In[58]:


# full=full.withColumn("item_count",col("item_count").cast(StringType()))
# full=full.withColumn("grand_total",col("grand_total").cast(StringType()))
# full=full.withColumn("payment_mode",col("payment_mode").cast(StringType()))
# full=full.withColumn("promo_code",col("promo_code").cast(StringType()))
# full=full.withColumn("vendor_discount_amount",col("vendor_discount_amount").cast(StringType()))
# full=full.withColumn("promo_code_discount_percentage",col("promo_code_discount_percentage").cast(StringType()))
# full=full.withColumn("is_favorite",col("is_favorite").cast(StringType()))
# full=full.withColumn("is_rated",col("is_rated").cast(StringType()))
# full=full.withColumn("driver_rating",col("driver_rating").cast(StringType()))
# full=full.withColumn("deliverydistance",col("deliverydistance").cast(StringType()))
# full=full.withColumn("preparationtime",col("preparationtime").cast(StringType()))
# full=full.withColumn("delivery_charge",col("delivery_charge").cast(StringType()))
# full=full.withColumn("serving_distance",col("serving_distance").cast(StringType()))
# full=full.withColumn("prepration_time",col("prepration_time").cast(StringType()))
# full=full.withColumn("discount_percentage",col("discount_percentage").cast(StringType()))
# full=full.withColumn("Free Delivery",col("Free Delivery").cast(StringType()))


# In[59]:


full=full.withColumnRenamed('target', 'label')


# In[60]:


# Write to full final
full.write.option("header","true").csv(r'./full_final_full.csv')


# In[ ]:




