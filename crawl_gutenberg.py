import requests
import bs4

import numpy as np

def getpagetext(parsedpage):
    # Remove HTML elements that are scripts
    scriptelements=parsedpage.find_all('script')
    # Concatenate the text content from all table cells
    for scriptelement in scriptelements:
        # Extract this script element from the page.
        # This changes the page given to this function!
        scriptelement.extract()
    pagetext=parsedpage.get_text()
    return(pagetext)

def get_book_text_url(url):

    # Find book text url
    cur_pagetocrawl_html=requests.get(url)
    cur_pagetocrawl_parsed=bs4.BeautifulSoup(cur_pagetocrawl_html.content,'html.parser')
    book_urls = cur_pagetocrawl_parsed.find("table", {"class": "files"})

    # Find elements that are hyperlinks
    pagelinkelements=book_urls.find_all('a')
    pageurls=[]
    for pagelink in pagelinkelements:
        pageurl_isok=1
        try:
            pageurl=pagelink['href']
        except:
            pageurl_isok=0
        if pageurl_isok==1:
            # Check that the url does NOT contain these strings
            if (pageurl.find('.pdf')!=-1)|(pageurl.find('.ps')!=-1):
                pageurl_isok=0
            # Check that the url DOES contain these strings
            if (pageurl.find('txt')==-1):
                pageurl_isok=0
            
        if pageurl_isok==1:
            return pageurl
    return pageurls

def get_book_text(txt_url):
    
    # Get page text
    pagetocrawl_html=requests.get(txt_url)
    pagetocrawl_parsed=bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
    page_text = getpagetext(pagetocrawl_parsed)

    start_str = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_str = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_ind = page_text[page_text.find(start_str):].find('\n') + page_text.find(start_str)
    end_ind = page_text.find(end_str)

    if(start_ind == -1 or end_ind == -1):
        start_str = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_str = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        start_ind = page_text[page_text.find(start_str):].find('\n') + page_text.find(start_str)
        end_ind = page_text.find(end_str)


    book = page_text[start_ind:end_ind]

    return book

def download_books(book_names, book_texts):
    # Saves the eBooks as txt files
    for name, text in zip(book_names, book_texts):
        np.savetxt(name, text)
