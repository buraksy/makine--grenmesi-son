"""
PREMIUM UI COMPONENTS
======================
World-class UI components for Streamlit
Inspired by: Stripe, Linear, Notion, Vercel, Framer
"""

def get_premium_css(theme='dark'):
    """
    Generate premium CSS with glassmorphism, animations, and modern design tokens
    """
    
    # Design Tokens
    if theme == 'dark':
        tokens = {
            # Colors
            'primary': '#1DB954',
            'primary_hover': '#1ED760',
            'primary_rgb': '29, 185, 84',
            'secondary': '#0A0A0A',
            'accent': '#FF6B6B',
            'warning': '#F59E0B',
            'success': '#10B981',
            'info': '#3B82F6',
            
            # Backgrounds
            'bg_primary': '#000000',
            'bg_secondary': '#0A0A0A',
            'bg_tertiary': '#141414',
            'bg_elevated': '#1A1A1A',
            'bg_overlay': 'rgba(10, 10, 10, 0.85)',
            
            # Glass
            'glass_bg': 'rgba(20, 20, 20, 0.6)',
            'glass_border': 'rgba(255, 255, 255, 0.1)',
            'glass_hover': 'rgba(20, 20, 20, 0.8)',
            
            # Text
            'text_primary': '#FFFFFF',
            'text_secondary': '#B3B3B3',
            'text_tertiary': '#757575',
            'text_muted': '#4A4A4A',
            
            # Borders
            'border_primary': 'rgba(255, 255, 255, 0.1)',
            'border_hover': 'rgba(29, 185, 84, 0.5)',
            'border_focus': '#1DB954',
            
            # Shadows
            'shadow_sm': '0 2px 8px rgba(0, 0, 0, 0.3)',
            'shadow_md': '0 4px 16px rgba(0, 0, 0, 0.4)',
            'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.5)',
            'shadow_xl': '0 16px 48px rgba(0, 0, 0, 0.6)',
            'shadow_glow': '0 0 32px rgba(29, 185, 84, 0.3)',
        }
    else:
        tokens = {
            'primary': '#1DB954',
            'primary_hover': '#1ED760',
            'primary_rgb': '29, 185, 84',
            'secondary': '#F8F9FA',
            'accent': '#FF6B6B',
            'warning': '#F59E0B',
            'success': '#10B981',
            'info': '#3B82F6',
            
            'bg_primary': '#FFFFFF',
            'bg_secondary': '#F8F9FA',
            'bg_tertiary': '#F1F3F5',
            'bg_elevated': '#FFFFFF',
            'bg_overlay': 'rgba(248, 249, 250, 0.95)',
            
            'glass_bg': 'rgba(255, 255, 255, 0.7)',
            'glass_border': 'rgba(0, 0, 0, 0.1)',
            'glass_hover': 'rgba(255, 255, 255, 0.9)',
            
            'text_primary': '#1A1A1A',
            'text_secondary': '#4A4A4A',
            'text_tertiary': '#757575',
            'text_muted': '#B3B3B3',
            
            'border_primary': 'rgba(0, 0, 0, 0.1)',
            'border_hover': 'rgba(29, 185, 84, 0.5)',
            'border_focus': '#1DB954',
            
            'shadow_sm': '0 2px 8px rgba(0, 0, 0, 0.05)',
            'shadow_md': '0 4px 16px rgba(0, 0, 0, 0.08)',
            'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.12)',
            'shadow_xl': '0 16px 48px rgba(0, 0, 0, 0.15)',
            'shadow_glow': '0 0 32px rgba(29, 185, 84, 0.2)',
        }
    
    return f"""
    <style>
    /* ============================================
       RESET & BASE STYLES
    ============================================ */
    
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    :root {{
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
        --spacing-2xl: 48px;
        --spacing-3xl: 64px;
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-full: 9999px;
        
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    /* ============================================
       MAIN LAYOUT
    ============================================ */
    
    .main {{
        background: {tokens['bg_primary']};
        color: {tokens['text_primary']};
    }}
    
    .block-container {{
        padding-top: var(--spacing-2xl) !important;
        padding-bottom: var(--spacing-3xl) !important;
        max-width: 1600px !important;
    }}
    
    /* ============================================
       HERO SECTION
    ============================================ */
    
    .premium-hero {{
        position: relative;
        background: linear-gradient(135deg, 
            rgba({tokens['primary_rgb']}, 0.95) 0%, 
            rgba({tokens['primary_rgb']}, 0.85) 50%,
            rgba({tokens['primary_rgb']}, 0.75) 100%);
        border-radius: var(--radius-xl);
        padding: var(--spacing-3xl);
        margin-bottom: var(--spacing-2xl);
        overflow: hidden;
        box-shadow: {tokens['shadow_xl']}, {tokens['shadow_glow']};
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .premium-hero::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: heroRotate 20s linear infinite;
    }}
    
    @keyframes heroRotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    .hero-content {{
        position: relative;
        z-index: 1;
    }}
    
    .hero-title {{
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin-bottom: var(--spacing-md);
        letter-spacing: -0.02em;
        text-shadow: 0 2px 16px rgba(0, 0, 0, 0.2);
    }}
    
    .hero-subtitle {{
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
        max-width: 800px;
        text-shadow: 0 1px 8px rgba(0, 0, 0, 0.15);
    }}
    
    .hero-badges {{
        display: flex;
        gap: var(--spacing-md);
        margin-top: var(--spacing-lg);
        flex-wrap: wrap;
    }}
    
    .hero-badge {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: var(--spacing-sm) var(--spacing-lg);
        border-radius: var(--radius-full);
        color: white;
        font-size: 0.875rem;
        font-weight: 600;
        transition: all var(--transition-base);
    }}
    
    .hero-badge:hover {{
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }}
    
    /* ============================================
       GLASS CARDS
    ============================================ */
    
    .glass-card {{
        background: {tokens['glass_bg']};
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid {tokens['glass_border']};
        border-radius: var(--radius-lg);
        padding: var(--spacing-xl);
        box-shadow: {tokens['shadow_md']};
        transition: all var(--transition-base);
    }}
    
    .glass-card:hover {{
        background: {tokens['glass_hover']};
        border-color: {tokens['border_hover']};
        box-shadow: {tokens['shadow_lg']};
        transform: translateY(-4px);
    }}
    
    /* ============================================
       KPI CARDS
    ============================================ */
    
    .kpi-card {{
        background: {tokens['bg_elevated']};
        border: 1px solid {tokens['border_primary']};
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        position: relative;
        overflow: hidden;
        transition: all var(--transition-base);
    }}
    
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {tokens['primary']} 0%, {tokens['primary_hover']} 100%);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform var(--transition-base);
    }}
    
    .kpi-card:hover {{
        border-color: {tokens['border_hover']};
        box-shadow: {tokens['shadow_lg']};
        transform: translateY(-4px);
    }}
    
    .kpi-card:hover::before {{
        transform: scaleX(1);
    }}
    
    .kpi-icon {{
        width: 48px;
        height: 48px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, {tokens['primary']} 0%, {tokens['primary_hover']} 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: var(--spacing-md);
        box-shadow: 0 4px 16px rgba({tokens['primary_rgb']}, 0.3);
    }}
    
    .kpi-value {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {tokens['text_primary']};
        line-height: 1;
        margin-bottom: var(--spacing-sm);
        background: linear-gradient(135deg, {tokens['primary']} 0%, {tokens['primary_hover']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .kpi-label {{
        font-size: 0.875rem;
        color: {tokens['text_secondary']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .kpi-change {{
        font-size: 0.875rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: var(--radius-sm);
        display: inline-block;
        margin-top: var(--spacing-sm);
    }}
    
    .kpi-change.positive {{
        color: {tokens['success']};
        background: rgba(16, 185, 129, 0.1);
    }}
    
    .kpi-change.negative {{
        color: {tokens['accent']};
        background: rgba(255, 107, 107, 0.1);
    }}
    
    /* ============================================
       PREDICTION RESULT CARD
    ============================================ */
    
    .prediction-card {{
        background: {tokens['bg_elevated']};
        border: 2px solid {tokens['border_primary']};
        border-radius: var(--radius-xl);
        padding: var(--spacing-2xl);
        position: relative;
        overflow: hidden;
        box-shadow: {tokens['shadow_xl']};
    }}
    
    .prediction-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, 
            {tokens['primary']} 0%, 
            {tokens['primary_hover']} 50%, 
            {tokens['success']} 100%);
    }}
    
    .prediction-score {{
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, {tokens['primary']} 0%, {tokens['primary_hover']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: var(--spacing-md);
        text-shadow: 0 4px 24px rgba({tokens['primary_rgb']}, 0.3);
    }}
    
    .prediction-label {{
        font-size: 1.25rem;
        color: {tokens['text_secondary']};
        font-weight: 600;
        margin-bottom: var(--spacing-lg);
    }}
    
    .prediction-confidence {{
        display: inline-flex;
        align-items: center;
        gap: var(--spacing-sm);
        background: rgba({tokens['primary_rgb']}, 0.1);
        border: 1px solid rgba({tokens['primary_rgb']}, 0.3);
        padding: var(--spacing-sm) var(--spacing-md);
        border-radius: var(--radius-full);
        font-size: 0.875rem;
        font-weight: 600;
        color: {tokens['primary']};
    }}
    
    /* ============================================
       BUTTONS
    ============================================ */
    
    .stButton > button {{
        background: linear-gradient(135deg, {tokens['primary']} 0%, {tokens['primary_hover']} 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: var(--spacing-md) var(--spacing-xl);
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba({tokens['primary_rgb']}, 0.3);
        transition: all var(--transition-base);
        cursor: pointer;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 8px 24px rgba({tokens['primary_rgb']}, 0.4);
        transform: translateY(-2px);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* ============================================
       SIDEBAR
    ============================================ */
    
    [data-testid="stSidebar"] {{
        background: {tokens['bg_secondary']};
        border-right: 1px solid {tokens['border_primary']};
    }}
    
    [data-testid="stSidebar"] .sidebar-content {{
        padding: var(--spacing-xl);
    }}
    
    .sidebar-logo {{
        display: flex;
        align-items: center;
        gap: var(--spacing-md);
        margin-bottom: var(--spacing-2xl);
        padding-bottom: var(--spacing-xl);
        border-bottom: 1px solid {tokens['border_primary']};
    }}
    
    .sidebar-logo-icon {{
        font-size: 2rem;
    }}
    
    .sidebar-logo-text {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {tokens['text_primary']};
    }}
    
    .sidebar-section {{
        margin-bottom: var(--spacing-xl);
    }}
    
    .sidebar-section-title {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {tokens['text_tertiary']};
        margin-bottom: var(--spacing-md);
    }}
    
    .sidebar-nav-item {{
        display: flex;
        align-items: center;
        gap: var(--spacing-md);
        padding: var(--spacing-md) var(--spacing-lg);
        border-radius: var(--radius-md);
        color: {tokens['text_secondary']};
        transition: all var(--transition-fast);
        cursor: pointer;
        margin-bottom: var(--spacing-xs);
    }}
    
    .sidebar-nav-item:hover {{
        background: {tokens['bg_tertiary']};
        color: {tokens['text_primary']};
    }}
    
    .sidebar-nav-item.active {{
        background: rgba({tokens['primary_rgb']}, 0.15);
        color: {tokens['primary']};
        font-weight: 600;
    }}
    
    /* ============================================
       FOOTER
    ============================================ */
    
    .premium-footer {{
        background: {tokens['glass_bg']};
        backdrop-filter: blur(20px) saturate(180%);
        border-top: 1px solid {tokens['glass_border']};
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
        padding: var(--spacing-2xl);
        margin-top: var(--spacing-3xl);
        box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.1);
    }}
    
    .footer-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: var(--spacing-2xl);
        margin-bottom: var(--spacing-xl);
    }}
    
    .footer-column h4 {{
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {tokens['text_primary']};
        margin-bottom: var(--spacing-md);
    }}
    
    .footer-column ul {{
        list-style: none;
        padding: 0;
    }}
    
    .footer-column li {{
        margin-bottom: var(--spacing-sm);
    }}
    
    .footer-column a {{
        color: {tokens['text_secondary']};
        text-decoration: none;
        transition: color var(--transition-fast);
        font-size: 0.875rem;
    }}
    
    .footer-column a:hover {{
        color: {tokens['primary']};
    }}
    
    .footer-bottom {{
        padding-top: var(--spacing-xl);
        border-top: 1px solid {tokens['border_primary']};
        text-align: center;
        color: {tokens['text_tertiary']};
        font-size: 0.875rem;
    }}
    
    /* ============================================
       ANIMATIONS
    ============================================ */
    
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{
            opacity: 1;
        }}
        50% {{
            opacity: 0.6;
        }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.5s ease-out;
    }}
    
    .animate-slide-in {{
        animation: slideIn 0.5s ease-out;
    }}
    
    /* ============================================
       LOADING SKELETON
    ============================================ */
    
    .skeleton {{
        background: linear-gradient(
            90deg,
            {tokens['bg_tertiary']} 0%,
            {tokens['bg_elevated']} 50%,
            {tokens['bg_tertiary']} 100%
        );
        background-size: 200% 100%;
        animation: skeleton-loading 1.5s ease-in-out infinite;
        border-radius: var(--radius-md);
    }}
    
    @keyframes skeleton-loading {{
        0% {{
            background-position: 200% 0;
        }}
        100% {{
            background-position: -200% 0;
        }}
    }}
    
    /* ============================================
       CHARTS
    ============================================ */
    
    .plotly-graph-div {{
        border-radius: var(--radius-lg);
        overflow: hidden;
    }}
    
    /* ============================================
       RESPONSIVE
    ============================================ */
    
    @media (max-width: 768px) {{
        .hero-title {{
            font-size: 2rem;
        }}
        
        .hero-subtitle {{
            font-size: 1rem;
        }}
        
        .prediction-score {{
            font-size: 3rem;
        }}
        
        .kpi-value {{
            font-size: 1.75rem;
        }}
        
        .footer-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    /* ============================================
       STREAMLIT OVERRIDES
    ============================================ */
    
    .stMarkdown {{
        color: {tokens['text_primary']};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {tokens['text_primary']};
        font-weight: 700;
    }}
    
    label {{
        color: {tokens['text_primary']} !important;
        font-weight: 500;
    }}
    
    .stSelectbox [data-baseweb="select"] {{
        background: {tokens['bg_elevated']};
        border-color: {tokens['border_primary']};
        border-radius: var(--radius-md);
    }}
    
    .stSlider {{
        padding-top: var(--spacing-md);
    }}
    
    .stProgress > div > div {{
        background: {tokens['primary']};
    }}
    
    /* ============================================
       METRIC ENHANCEMENTS
    ============================================ */
    
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {tokens['primary']};
    }}
    
    [data-testid="stMetricDelta"] {{
        font-size: 0.875rem;
        font-weight: 600;
    }}
    
    /* ============================================
       EXPANDER STYLING
    ============================================ */
    
    .streamlit-expanderHeader {{
        background: {tokens['bg_elevated']};
        border: 1px solid {tokens['border_primary']};
        border-radius: var(--radius-md);
        font-weight: 600;
        color: {tokens['text_primary']};
        transition: all var(--transition-fast);
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {tokens['border_hover']};
        background: {tokens['bg_tertiary']};
    }}
    
    </style>
    """


def render_hero_section():
    """Render premium hero section"""
    import streamlit as st
    
    st.markdown("""
    <div class="premium-hero animate-fade-in">
        <div class="hero-content">
            <h1 class="hero-title">🎵 Spotify Popularity AI</h1>
            <p class="hero-subtitle">
                World-class machine learning platform for predicting song popularity. 
                Powered by advanced ensemble learning and explainable AI.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">✨ Production Ready</span>
                <span class="hero-badge">🎯 R² = 0.5694</span>
                <span class="hero-badge">🚀 114K Songs Trained</span>
                <span class="hero-badge">🔬 SHAP Explainable</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render premium footer"""
    import streamlit as st
    from datetime import datetime
    
    current_year = datetime.now().year
    
    st.markdown(f"""
    <div class="premium-footer animate-fade-in">
        <div class="footer-grid">
            <div class="footer-column">
                <h4>🎵 Spotify Popularity AI</h4>
                <p style="color: #B3B3B3; font-size: 0.875rem; margin-top: 8px;">
                    Advanced machine learning platform for music analytics and popularity prediction.
                </p>
            </div>
            
            <div class="footer-column">
                <h4>Navigation</h4>
                <ul>
                    <li><a href="#dashboard">📊 Dashboard</a></li>
                    <li><a href="#analytics">📈 Analytics</a></li>
                    <li><a href="#predictions">🎯 Predictions</a></li>
                    <li><a href="#insights">💡 Insights</a></li>
                </ul>
            </div>
            
            <div class="footer-column">
                <h4>Technology Stack</h4>
                <ul>
                    <li><a href="#">Streamlit</a></li>
                    <li><a href="#">Python 3.14</a></li>
                    <li><a href="#">Scikit-Learn</a></li>
                    <li><a href="#">SHAP</a></li>
                    <li><a href="#">Plotly</a></li>
                </ul>
            </div>
            
            <div class="footer-column">
                <h4>Model Info</h4>
                <ul>
                    <li><a href="#">BaggingRegressor</a></li>
                    <li><a href="#">R² Score: 0.5694</a></li>
                    <li><a href="#">RMSE: 14.58</a></li>
                    <li><a href="#">MAE: 10.67</a></li>
                </ul>
            </div>
        </div>
        
        <div class="footer-bottom">
            <p>© {current_year} Spotify Popularity AI • All Rights Reserved • Built with ❤️ and 🤖</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_kpi_card(icon, value, label, change=None, change_type="positive"):
    """Create a premium KPI card"""
    change_html = ""
    if change:
        change_class = "positive" if change_type == "positive" else "negative"
        change_symbol = "↑" if change_type == "positive" else "↓"
        change_html = f'<div class="kpi-change {change_class}">{change_symbol} {change}</div>'
    
    return f"""
    <div class="kpi-card animate-fade-in">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {change_html}
    </div>
    """


def create_prediction_card(score, confidence, label="Tahmin Edilen Popülerlik"):
    """Create a premium prediction result card"""
    return f"""
    <div class="prediction-card animate-fade-in">
        <div class="prediction-score">{score}</div>
        <div class="prediction-label">{label}</div>
        <div class="prediction-confidence">
            <span>🎯</span>
            <span>Güven: {confidence}%</span>
        </div>
    </div>
    """


def create_glass_card(content):
    """Create a glassmorphism card"""
    return f"""
    <div class="glass-card animate-fade-in">
        {content}
    </div>
    """
