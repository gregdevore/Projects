# Web scraping example

library(rvest)
library(RSelenium)
library(XML)
library(dplyr)
library(tidyr)
library(ggplot2)
# Start selenium server and browser
rsDriver()
remDr <- remoteDriver(remoteServerAddr = "localhost", port = 4445L, browserName = "chrome")
remDr$open()

# Start at base URL
url <- 'http://www.marathonguide.com/results/browse.cfm?MIDD=472171105'
p <- html_session(url)
# Store yearly results in list (may have different number of columns depending on the year)
overall_results <- list()
i <- 1
gender <- 'Women'
years <- c(seq(2000,2011),seq(2013,2017))
for (year in years) {
    cat('Scraping year ',year,' for ',gender,'\'s results\n', sep = '')
    pi <- follow_link(p,as.character(year))
    # Navigate to page
    remDr$navigate(pi$url)
    
    # Select proper results dropdown and top 100 finishers
    if (gender == 'Men') {
      css <- 'select+ p select'
    } else {
      css <- 'p+ p select'
    }
    raceRange <- remDr$findElement(using = 'css selector', css)      
    raceRange$clickElement()
    if (gender == 'Men') {
      xpath <- '/html/body/table[2]/tbody/tr[1]/td[2]/table[3]/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr/td/table[2]/tbody/tr/td[1]/form/p[1]/select/option[2]'
    } else {
      xpath <- '/html/body/table[2]/tbody/tr[1]/td[2]/table[3]/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr/td/table[2]/tbody/tr/td[1]/form/p[2]/select/option[2]'
    }
    raceRangeValue <- remDr$findElement(using = 'xpath', xpath)
    raceRangeValue$clickElement()
    
    # Click 'View' to see race results
    viewButton <- remDr$findElement(using = 'xpath', '/html/body/table[2]/tbody/tr[1]/td[2]/table[3]/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr/td/table[2]/tbody/tr/td[1]/form/p[3]/input[3]')
    viewButton$clickElement()
    # Give page time to load
    Sys.sleep(2)
    
    # Grab HTML table and parse into data frame
    table <- remDr$findElement(using = 'xpath','/html/body/table[2]/tbody/tr[1]/td[2]/table[3]/tbody/tr[2]/td/table')
    elemtxt <- table$getElementAttribute('outerHTML')
    elemxml <- htmlTreeParse(elemtxt[[1]], useInternalNodes=T)
    results <- readHTMLTable(elemxml)[[1]]
    # Extract row 5 (header data)
    col.names <- unlist(results[5,])
    names(results) <- col.names
    # Remove rows 1 - 5 (not relevant)
    results <- results[-c(seq(1,5)), ]  
    # Store in list
    overall_results[[i]] <- results
    i <- i + 1    
}

# Parse results, add to single data frame with consistent column names
results.df <- data.frame(Name = c())
for (ind in seq(1,length(overall_results))) {
  df <- overall_results[[ind]]
  names <- grep('name',names(df),ignore.case = TRUE)
  time <- grep('time',names(df),ignore.case = TRUE)
  country <- grep('country',names(df),ignore.case = TRUE)
  colnames(df)[c(names,time[1],country)] <- c('Name','Time','Country')
  df[c(names,time[1],country)] <- lapply(df[c(names,time[1],country)], as.character)
  df$Year <- years[ind]
  results.df <- rbind(results.df, df[, c(names,time[1],country,ncol(df))])
}

# Split time into hour/minute/second, convert to hours
results.df <- results.df %>% separate(Time, c('Hour','Minute','Second'),':')
results.df[, c('Hour','Minute','Second')] <- lapply(results.df[, c('Hour','Minute','Second')],as.numeric)
results.df <- results.df %>% mutate(TotalTime = Hour + Minute/60 + Second/3600)

# Compute time stats for each year
results.timeSummary <- results.df %>% group_by(Year) %>% 
  summarize(bestTime = min(TotalTime), worstTime = max(TotalTime), meanTime = mean(TotalTime),
            sd = sd(TotalTime))

# Plot best times
ggplot(results.timeSummary, aes(x = Year, y = bestTime)) + geom_line() + 
  ylim(2.0,2.5) + ylab('Time (Hours)') + xlab('Year') + ggtitle(sprintf('%s\'s Best Finishing Time',gender))
# Plot average time with error bars (1 SD)
ggplot(results.timeSummary, aes(x = Year, y = meanTime)) +
  geom_ribbon(aes(ymin = meanTime - sd, ymax = meanTime + sd), fill = "grey70") + 
  geom_line() + ylab('Time (Hours)') + xlab('Year') + 
  ggtitle(sprintf('%s\'s Mean Finishing Time With SD Bars',gender)) +
  ylim(2,3.5)

# Plot distribution of finishing times for each year
minT <- min(results.df$TotalTime)
maxT <- max(results.df$TotalTime)
ggplot(results.df %>% filter(Year < 2004), aes(TotalTime)) + geom_density() + 
  facet_grid(Year ~ .) + xlim(minT, maxT) + ylim(0, 6) + 
  ggtitle(sprintf('%s\'s Finishing Times By Year',gender))
ggplot(results.df %>% filter(Year >= 2004 & Year < 2008), aes(TotalTime)) + geom_density() + 
  facet_grid(Year ~ .) + xlim(minT, maxT) + ylim(0, 6)
ggplot(results.df %>% filter(Year >= 2008 & Year < 2012), aes(TotalTime)) + geom_density() + 
  facet_grid(Year ~ .) + xlim(minT, maxT) + ylim(0, 6)
ggplot(results.df %>% filter(Year >= 2012 & Year < 2018), aes(TotalTime)) + geom_density() + 
  facet_grid(Year ~ .) + xlim(minT, maxT) + ylim(0, 6)
