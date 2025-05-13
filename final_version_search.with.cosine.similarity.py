import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Book Search Engine",
    page_icon="📚",
    layout="wide"
)

# Define the path to your book covers folder
BOOK_COVERS_PATH = r"C:\Users\User\Desktop\big data project\Book Covers of Amazon Data & AI Books"

# Function to load data and prepare TF-IDF
@st.cache_data
def load_data_and_prepare_tfidf():
    # Load the dataset
    file_path = r"C:\Users\User\Desktop\big data project\books_search_ready.csv"
    df = pd.read_csv(file_path)
    
    # Make sure search_blob column exists, otherwise create it
    if 'search_blob' not in df.columns:
        df['search_blob'] = df.apply(build_blob, axis=1)
    
    # Create TF-IDF vectorizer and transform search_blob column
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_blob'])
    
    return df, vectorizer, tfidf_matrix

# Function to build search blob from row data
def build_blob(row):
    return f"{row['title']} {row['author']} {row['language']} {row['publisher']} " \
           f"{row['cluster_name']} rating {row['avg_reviews']} stars {row['mean_rating']} " \
           f"{int(row['n_reviews'])} reviews priced at ${row['price']:.2f} " \
           f"{int(row['pages'])} pages"

# Function to search books using TF-IDF and cosine similarity
def search_books_tfidf(query, df, vectorizer, tfidf_matrix, top_n=5):
    if not query:
        return pd.DataFrame()
    
    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get top books with similarity scores
    top_books = df.iloc[top_indices].copy()
    
    # Add similarity scores (0-100%)
    if not top_books.empty:
        top_books['similarity'] = similarities[top_indices]
        top_books['similarity_pct'] = top_books['similarity'] * 100
    
    return top_books

# Function to display book details with local image
def display_book_details(book):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Try to load and display local book image
        try:
            if pd.notna(book['matched_image']):
                # Get image path from matched_image column
                image_filename = book['matched_image']
                image_path = os.path.join(BOOK_COVERS_PATH, image_filename)
                
                # Check if file exists
                if os.path.isfile(image_path):
                    img = Image.open(image_path)
                    st.image(img, width=200)
                else:
                    st.image("https://via.placeholder.com/200x300?text=Cover+Not+Found", width=200)
                    st.caption(f"Image not found: {image_filename}")
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Cover+Available", width=200)
        except Exception as e:
            st.image("https://via.placeholder.com/200x300?text=Error+Loading+Image", width=200)
            st.caption(f"Error: {str(e)}")
        
        # Display ratings visualization
        st.subheader("Ratings Distribution")
        ratings_data = {
            '5★': book['star5'],
            '4★': book['star4'],
            '3★': book['star3'],
            '2★': book['star2'],
            '1★': book['star1']
        }
        
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(ratings_data.keys(), ratings_data.values(), color=sns.color_palette("YlOrRd", 5))
        ax.set_ylabel('Number of Reviews')
        ax.set_title(f'Review Distribution (Avg: {book["mean_rating"]:.2f}/5)')
        
        # Add percentage labels on bars
        total_reviews = sum(ratings_data.values())
        if total_reviews > 0:
            for bar in bars:
                height = bar.get_height()
                percentage = height / total_reviews * 100
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
    
    with col2:
        # Show match percentage if available
        if 'similarity_pct' in book:
            st.metric("Match Score", f"{book['similarity_pct']:.1f}%")
        
        st.title(book['title'])
        st.subheader(f"by {book['author']}")
        
        st.markdown(f"""
        **Price:** ${book['price']:.2f}  
        **Publisher:** {book['publisher']}  
        **Language:** {book['language']}  
        **Pages:** {int(book['pages']) if not pd.isna(book['pages']) else 'N/A'}  
        **ISBN-13:** {book['ISBN_13']}  
        **Dimensions:** {book['dimensions']}  
        **Weight:** {book['weight']} pounds  
        **Category:** {book['cluster_name']}  
        """)
        
        # Display review stats
        st.markdown(f"""
        ### Reviews
        **Average Rating:** {book['avg_reviews']:.2f}/5 ({int(book['n_reviews'])} reviews)
        """)
        
        # Link to book
        if pd.notna(book['link']) and book['link']:
            st.markdown(f"[View on Amazon]({book['complete_link']})")

def main():
    # Load data and prepare TF-IDF
    try:
        df, vectorizer, tfidf_matrix = load_data_and_prepare_tfidf()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Page title and description
    st.title("📚 Book Search Engine")
    st.markdown("Search for books using TF-IDF semantic search")
    
    # Main search interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Search box
        st.subheader("Search")
        query = st.text_input("Enter search terms:", placeholder="Example: machine learning")
        
        num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)
        
        st.divider()
        
        # Basic filters
        st.subheader("Quick Filters")
        
        price_range = st.slider(
            "Price Range ($):", 
            min_value=float(df['price'].min()),
            max_value=float(df['price'].max()),
            value=(float(df['price'].min()), float(df['price'].max()))
        )
        
        min_rating = st.slider("Minimum Rating:", 1.0, 5.0, 1.0, 0.1)
        
        languages = sorted(df['language'].unique())
        selected_language = st.selectbox("Language:", ['All'] + list(languages))
        
        # Search button
        search_pressed = st.button("Search", type="primary")
    
    # Display search results
    with col2:
        if query or search_pressed:
            # Perform TF-IDF search
            results = search_books_tfidf(query, df, vectorizer, tfidf_matrix, num_results)
            
            # Apply filters
            if not results.empty:
                # Price filter
                results = results[(results['price'] >= price_range[0]) & 
                                  (results['price'] <= price_range[1])]
                
                # Rating filter
                results = results[results['avg_reviews'] >= min_rating]
                
                # Language filter
                if selected_language != 'All':
                    results = results[results['language'] == selected_language]
            
            # Show results
            if not results.empty:
                st.success(f"Found {len(results)} books matching '{query}'")
                
                # Display results as expandable sections
                for i, (_, book) in enumerate(results.iterrows()):
                    with st.expander(
                        f"{i+1}. {book['title']} ({book['similarity_pct']:.1f}% match)"
                    ):
                        display_book_details(book)
            else:
                st.warning(f"No books found matching '{query}' with the current filters.")
        else:
            # Show random recommendations when no search is performed
            st.subheader("Random Book Recommendations")
            random_books = df.sample(min(5, len(df)))
            
            for i, (_, book) in enumerate(random_books.iterrows()):
                with st.expander(f"Book {i+1}: {book['title']}"):
                    display_book_details(book)

if __name__ == "__main__":
    main()