use test

//Single Grouping

db.products.aggregate([
    {$group:
     {
	 _id:"$manufacturer", 
	 num_products:{$sum:1}
     }
    }
])

//Single Grouping with renaming the _id. Easier to Read.
db.products.aggregate([
    {$group:
     {
	 _id: {'manufacturer':"$manufacturer"},
	 num_products:{$sum:1}
     }
    }
])

//Double Grouping. Compound Key. 

db.products.aggregate([
    {$group:
     {
     _id: {
         "maker":"$manufacturer", 
         "category" : "$category"},
     avg_price:{$avg:"$price"}
     }
    }
])
