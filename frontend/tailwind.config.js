/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['GreycliffCF', 'Kumbh Sans', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      colors: {
        primary: {
          50:  '#EEEFFE',
          100: '#DDDFFE',
          200: '#BCBDFB',
          300: '#9A9CF5',
          400: '#7E7FEB',
          500: '#6667E0',
          600: '#5152D5',
          700: '#4041C0',
          800: '#3335A0',
          900: '#1E1F7A',
        },
        // Blossom's navy-tinted gray scale (matches BlossomOLBAdminFront)
        gray: {
          50:  '#f8fafc',
          100: '#f3f6f9',
          200: '#e0e6ed',
          300: '#bcccde',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#002855',
          800: '#001d3d',
          900: '#000d1c',
        },
      },
      boxShadow: {
        sm:  '0 1px 3px rgba(0, 29, 61, 0.06)',
        DEFAULT: '0 2px 8px rgba(0, 29, 61, 0.08)',
        md:  '0 4px 16px rgba(0, 29, 61, 0.08)',
        lg:  '0 6px 36px rgba(0, 29, 61, 0.08)',
      },
    },
  },
  plugins: [],
}
