from icrawler.builtin import GoogleImageCrawler

keywords = ['car', 'suv', 'bmw car', 'vw car', 'landrover car', 'audi car', 'hyundai car']

for keyword in keywords:
  google_crawler = GoogleImageCrawler(
    parser_threads=3,
    downloader_threads=5,
    storage={'root_dir': 'CrawlImages'}
  )

  google_crawler.crawl(keyword=keyword, max_num=500, min_size=(400,400))
