/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      screens: {
        'xs': '480px',  // Extra small screen breakpoint
      },
      colors: {
        primary: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        forged: '#ef4444',
        genuine: '#22c55e',
        dark: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s infinite',
        'shimmer': 'shimmer 8s infinite linear',
        'float': 'float 6s ease-in-out infinite',
        'fadeIn': 'fadeIn 0.7s ease-out forwards',
        'bounce-subtle': 'bounce-subtle 3s ease-in-out infinite',
        'bounce-slow': 'bounce 3s infinite',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(16, 185, 129, 0.2)',
        'glow-lg': '0 0 30px rgba(16, 185, 129, 0.3)',
        'inner-glow': 'inset 0 0 20px rgba(16, 185, 129, 0.2)',
        'inner-lg': 'inset 0 2px 4px 0 rgb(0 0 0 / 0.15)',
      },
      transitionProperty: {
        'width': 'width',
        'height': 'height',
        'spacing': 'margin, padding',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: '65ch',
            color: 'inherit',
            a: {
              color: '#10b981',
              '&:hover': {
                color: '#34d399',
              },
            },
          },
        },
      },
    },
  },
  plugins: [],
  variants: {
    extend: {
      opacity: ['group-hover'],
      transform: ['group-hover'],
      scale: ['group-hover'],
      translate: ['group-hover'],
    },
  },
};