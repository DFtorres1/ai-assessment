interface BlossomMarkProps {
  size?: number
  className?: string
}

const PETAL_ANGLES = [0, 60, 120, 180, 240, 300]

const BlossomMark = ({ size = 32, className = 'text-primary-700' }: BlossomMarkProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 40 40"
    fill="none"
    className={className}
    aria-hidden
  >
    {PETAL_ANGLES.map(angle => (
      <ellipse
        key={angle}
        cx="20"
        cy="11.5"
        rx="3.8"
        ry="8"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        transform={`rotate(${angle} 20 20)`}
      />
    ))}
    <circle
      cx="20"
      cy="20"
      r="3.2"
      stroke="currentColor"
      strokeWidth="1.5"
    />
  </svg>
)

export default BlossomMark
