import cfscrape
import csv
import os
from lxml import etree
import xml.etree.ElementTree as ET

scraper = cfscrape.create_scraper()


# realnews
real_csv_file_path = "news_data.csv"

def nation_com_pk():

    url = 'https://www.nation.com.pk/sitemap.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://www.nation.com.pk/sitemap_2")]

    count = 1
    max_count = 3000

    for yl in year_links:
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)
            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://www.nation.com.pk")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text

                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='jeg_post_title']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("//div[contains(@class,'jeg_meta_date jeg_meta_date_dp')]")[0]
                        date_text = date_ele.text
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "//div[contains(@class,'entry-content')]//text()")
                    description_text = " ".join(description_eles).strip()

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'True']

                    file_exists = os.path.exists(real_csv_file_path)
                    with open(real_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Nation post error: ", e)
                    pass

        except Exception as e:
            print("Nation error: ", e)
            pass

def pakistantoday_com_pk():

    url = 'https://www.pakistantoday.com.pk/wp-sitemap.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://www.pakistantoday.com.pk/wp-sitemap-posts-post")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)
            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'

            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [loc.text for loc in post_links_ele if loc.text.startswith("https://www.pakistantoday.com.pk")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text

                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='tdb-title-text']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:

                        date_ele = tree.xpath("//div/time//text()")[0]
                        date_text = date_ele
                    except:
                        date_text=''

                    description_elements = tree.xpath(
                        "//div[@class='td_block_wrap tdb_single_content tdi_45 td-pb-border-top td_block_template_1 td-post-content tagdiv-type']/*[not(self::style)]//text()")
                    description_text = " ".join(description_elements).strip()

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'True']

                    file_exists = os.path.exists(real_csv_file_path)
                    with open(real_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Pakistan Today post error: ", e)
                    pass

        except Exception as e:
            print("Pakistan Today error: ", e)
            pass

def pakobserver_net():

    url = 'https://pakobserver.net/sitemap_index.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://pakobserver.net/post-sitemap")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://pakobserver.net/")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='jeg_post_title']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("(//div[@class='jeg_meta_date']/a//text())")[0]
                        date_text = date_ele
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "//div[contains(@class,'entry-content')]//text()")
                    description_text = " ".join(description_eles).strip().replace("\n", " ")

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'True']

                    file_exists = os.path.exists(real_csv_file_path)
                    with open(real_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Pakobserver post error: ", e)
                    pass

        except Exception as e:
            print("Pakobserver error: ", e)
            pass

def factly_in():

    url = 'https://factly.in/sitemap_index.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://factly.in/post-")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://factly.in/")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[contains(@class,'post-title')]")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("//span/time")[0]
                        date_text = date_ele.get('title')
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "///div[contains(@class,'post-content description')]//text()")
                    description_text = " ".join(description_eles).strip().replace("\n", " ")

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'True']

                    file_exists = os.path.exists(real_csv_file_path)
                    with open(real_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Factly post in error: ", e)
                    pass

        except Exception as e:
            print("Factly error: ", e)
            pass


#fakenews

fake_csv_file_path = "fake_news_data.csv"
def opindia_com():

    url = 'https://www.opindia.com/sitemap_index.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("http://www.opindia.com/post-")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://www.opindia.com/")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='tdb-title-text']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("//div/time[@class='entry-date updated td-module-date']")[0]
                        date_text = date_ele.get('datetime')
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "//div[contains(@class,'td_block_wrap tdb_single_content tdi_135 td-pb-border-top td_block_template_1 td-post-content tagdiv-type')]/*[not(self::style)]//text()")
                    description_text = " ".join(description_eles).strip().replace("\n", " ")

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'False']

                    file_exists = os.path.exists(fake_csv_file_path)
                    with open(fake_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Opindia post error: ", e)
                    pass

        except Exception as e:
            print("Opindia error: ", e)
            pass

def the_fauxy_com():
    url = 'https://thefauxy.com/sitemap_index.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://thefauxy.com/post-")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://thefauxy.com/")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='page-title']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text) <= 5:
                        continue

                    try:
                        date_ele = tree.xpath("(//li/time[@class='ct-meta-element-date'])[1]")[0]
                        date_text = date_ele.get('datetime')
                    except:
                        date_text = ''

                    description_eles = tree.xpath(
                        "//div[@class='entry-content']//text()")
                    description_text = " ".join(description_eles).strip()

                    if not description_text or len(description_text) <= 20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'False']

                    file_exists = os.path.exists(fake_csv_file_path)
                    with open(fake_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("fauxy post error: ", e)
                    pass

        except Exception as e:
            print("fauxy error: ", e)
            pass

def news_biscuit_com():
    url = 'https://www.newsbiscuit.com/blog-posts-sitemap.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'
    post_links_ele = root.findall('.//{}loc'.format(namespace))
    post_links = [link.text for link in post_links_ele if link.text.startswith("https://www.newsbiscuit.com/")]

    count = 1
    max_count = 3000
    for pl in post_links:
        if count >= max_count:
            break
        try:
            res = scraper.get(pl)

            html_content = res.text
            tree = etree.HTML(html_content)

            title_ele = tree.xpath("//span/span[contains(@class,'blog-post-title-font blog-post-title-color')]")[0]
            title_text = title_ele.text

            if not title_text or len(title_text) <= 5:
                continue

            try:
                date_ele = tree.xpath("//span[@class='post-metadata__date time-ago']")[0]
                date_text = date_ele.get('title')
            except:
                date_text = ''

            description_eles = tree.xpath(
                "//div[@class='KcNlj']//text()")
            description_text = " ".join(description_eles).strip()

            if not description_text or len(description_text) <= 20:
                continue

            data_to_append = [pl, title_text, description_text, date_text, 'False']

            file_exists = os.path.exists(fake_csv_file_path)
            with open(fake_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                csv_writer = csv.writer(file)
                # Write headers if the file doesn't exist
                if not file_exists:
                    csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                csv_writer.writerow(data_to_append)

            count += 1

        except Exception as e:
            print("New biscuit error: ", e)
            pass

def aninews_in():

    url = 'https://www.aninews.in/sitemap.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://www.aninews.in/sitemap-news.xml")]

    count = 1
    max_count = 3000

    for yl in year_links:
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://www.aninews.in/news")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='title']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("//article//div/p/span[@class='time-red']//text()")[0]
                        date_text = date_ele
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "//div[@class='content count-br']//text()")
                    description_text = " ".join(description_eles).strip().replace("\n", " ")

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'False']

                    file_exists = os.path.exists(fake_csv_file_path)
                    with open(fake_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Opindia post error: ", e)
                    pass

        except Exception as e:
            print("Opindia error: ", e)
            pass

def waron_fakes_com():

    url = 'https://waronfakes.com/sitemap_index.xml'
    res = scraper.get(url)

    root = ET.fromstring(res.text)
    namespace = root.tag.split('}')[0] + '}'

    year_links_ele = root.findall('.//{}loc'.format(namespace))
    year_links = [link.text for link in year_links_ele if link.text.startswith("https://waronfakes.com/post-sitemap")]

    count = 1
    max_count = 3000

    for yl in reversed(year_links):
        if count >= max_count:
            break
        try:
            res = scraper.get(yl)

            root = ET.fromstring(res.text)
            namespace = root.tag.split('}')[0] + '}'
            post_links_ele = root.findall('.//{}loc'.format(namespace))
            post_links = [link.text for link in post_links_ele if link.text.startswith("https://waronfakes.com/")]

            for pl in post_links:
                if count >= max_count:
                    break
                try:
                    res = scraper.get(pl)

                    html_content = res.text
                    tree = etree.HTML(html_content)

                    title_ele = tree.xpath("//h1[@class='entry-title']")[0]
                    title_text = title_ele.text

                    if not title_text or len(title_text)<=5:
                        continue

                    try:
                        date_ele = tree.xpath("//article/header/div/span[@class='entry-date']")[0]
                        date_text = date_ele.text
                    except:
                        date_text=''

                    description_eles = tree.xpath(
                        "//div[@class='entry-content']//text()")
                    description_text = " ".join(description_eles).strip().replace("\n", " ")

                    if not description_text or len(description_text)<=20:
                        continue

                    data_to_append = [pl, title_text, description_text, date_text, 'False']

                    print(data_to_append)

                    file_exists = os.path.exists(fake_csv_file_path)
                    with open(fake_csv_file_path, 'a', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        # Write headers if the file doesn't exist
                        if not file_exists:
                            csv_writer.writerow(["Links", "Title", "Description", "Date", "Status"])

                        csv_writer.writerow(data_to_append)

                    count += 1

                except Exception as e:
                    print("Waronfake post error: ", e)
                    pass

        except Exception as e:
            print("Waronfake error: ", e)
            pass


if __name__ == '__main__':
    # try:
    #     nation_com_pk()
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     pakobserver_net()
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     pakistantoday_com_pk()
    # except Exception as e:
    #     print(e)
    # try:
    #     factly_in()
    # except Exception as e:
    #     print(e)

    # try:
    #     opindia_com()
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     the_fauxy_com()
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     news_biscuit_com()
    # except Exception as e:
    #     print(e)

    # try:
    #     aninews_in()
    # except Exception as e:
    #     print(e)

    try:
        waron_fakes_com()
    except Exception as e:
        print(e)



