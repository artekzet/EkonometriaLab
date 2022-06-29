name := "ZaliczenieESI"

version := "0.1"

scalaVersion := "2.12.10"

resolvers ++= Seq(
  "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven",
  "Typesafe Simple Repository" at "https://repo.typesafe.com/typesafe/simple/maven-releases",
  "MavenRepository" at "https://mvnrepository.com"
)


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.0.2",
  "org.apache.spark" %% "spark-sql" % "3.0.2",
  "org.apache.spark" %% "spark-mllib" % "3.0.2",
)