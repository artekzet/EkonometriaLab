import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ZaliczenieEkonometria extends App {
  val spark: SparkSession = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val taksowki: DataFrame = spark.read.load("src/main/resources/NYC_taxi_2009-2016.parquet")

  taksowki.printSchema()

  taksowki.withColumn("hour_of_day", hour(col("pickup_datetime")))
    .groupBy("hour_of_day")
    .agg(avg("trip_distance").as("averageDistance"))
    .orderBy(col("averageDistance").desc_nulls_last)
    .show()

  taksowki.withColumn("zone", col("pickup_taxizone_id"))
    .groupBy("zone")
    .agg(avg("trip_distance").as("averageDistance"))
    .orderBy(col("averageDistance").desc_nulls_last)
    .show()

  taksowki.withColumn("zone", col("pickup_taxizone_id"))
    .withColumn("hour_of_day", hour(col("pickup_datetime")))
    .groupBy("zone", "hour_of_day")
    .agg(avg("trip_distance").as("averageDistance"), count("*").as("totalTrips"))
    .filter("totalTrips > 30")
    .orderBy(col("averageDistance").desc_nulls_last)

  val grupowaniePoGodzinieIMiejscu: DataFrame = taksowki.selectExpr("pickup_taxizone_id", "hour(pickup_datetime) as hour_of_day", "trip_distance")


  val strefyTaxi: DataFrame = spark.read
    .options(Map("header" -> "true", "inferSchema" -> "true"))
    .csv("src/main/resources/taxi_zones.csv")

  strefyTaxi.show()

  def przewidywanaDlugoscTrasy(strefa: Int, godzina: Int): Unit = {
    grupowaniePoGodzinieIMiejscu
      .filter(s"hour_of_day = $godzina")
      .filter(s"pickup_taxizone_id = $strefa")
      .agg(avg("trip_distance").as("averageDistance"))
      .show()
  }

  przewidywanaDlugoscTrasy(132, 7)

  grupowaniePoGodzinieIMiejscu.show()

//  grupowaniePoGodzinieIMiejscu.write
//    .option("header", value = true)
//    .csv("src/main/resources/hourAndPlace.csv")

  /**
   * Zakomentowane, żeby nazwa pliku używa w skrypcie Pythona była taka sama
   */
}
