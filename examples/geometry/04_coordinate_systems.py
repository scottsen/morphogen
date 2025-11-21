"""Coordinate System Conversions Example.

Demonstrates conversions between coordinate systems:
- Cartesian ↔ Polar (2D)
- Cartesian ↔ Spherical (3D)
- Practical applications
"""

import numpy as np
from morphogen.stdlib.geometry import (
    point2d, point3d,
    cartesian_to_polar, polar_to_cartesian,
    cartesian_to_spherical, spherical_to_cartesian,
    distance_point_point
)


def main():
    print("=" * 60)
    print("Coordinate System Conversions Example")
    print("=" * 60)

    # Cartesian to Polar
    print("\n1. Cartesian → Polar Conversion")
    print("-" * 40)

    points_cartesian = [
        point2d(3, 4),
        point2d(1, 1),
        point2d(-1, 0),
        point2d(0, 5),
    ]

    print("Converting Cartesian (x,y) to Polar (r,θ):\n")
    for p in points_cartesian:
        r, theta = cartesian_to_polar(p)
        theta_deg = theta * 180 / np.pi

        print(f"({p.x:6.2f}, {p.y:6.2f}) → "
              f"r={r:6.3f}, θ={theta:6.3f} rad ({theta_deg:6.1f}°)")

    # Polar to Cartesian
    print("\n2. Polar → Cartesian Conversion")
    print("-" * 40)

    polar_coords = [
        (5.0, 0.0),                # 0°
        (5.0, np.pi/4),            # 45°
        (5.0, np.pi/2),            # 90°
        (5.0, np.pi),              # 180°
        (5.0, 3*np.pi/2),          # 270°
    ]

    print("Converting Polar (r,θ) to Cartesian (x,y):\n")
    for r, theta in polar_coords:
        p = polar_to_cartesian(r, theta)
        theta_deg = theta * 180 / np.pi

        print(f"r={r:.1f}, θ={theta:6.3f} rad ({theta_deg:6.1f}°) → "
              f"({p.x:6.3f}, {p.y:6.3f})")

    # Roundtrip conversion
    print("\n3. Roundtrip Conversion (Cartesian → Polar → Cartesian)")
    print("-" * 40)

    original = point2d(7, 3)
    r, theta = cartesian_to_polar(original)
    converted = polar_to_cartesian(r, theta)

    print(f"Original:  ({original.x}, {original.y})")
    print(f"Polar:     r={r:.3f}, θ={theta:.3f}")
    print(f"Converted: ({converted.x:.3f}, {converted.y:.3f})")
    print(f"Error:     {abs(original.x - converted.x):.10f}")

    # Spiral generation using polar coordinates
    print("\n4. Generating Spiral Pattern (Polar)")
    print("-" * 40)

    print("Archimedean spiral: r = a + b*θ\n")

    a, b = 0.5, 0.3
    spiral_points = []

    for i in range(10):
        theta = i * np.pi / 4  # 45° increments
        r = a + b * theta

        p = polar_to_cartesian(r, theta)
        spiral_points.append(p)

        theta_deg = theta * 180 / np.pi
        print(f"θ={theta_deg:6.1f}°: r={r:.3f} → ({p.x:6.3f}, {p.y:6.3f})")

    # 3D: Cartesian to Spherical
    print("\n5. Cartesian → Spherical Conversion (3D)")
    print("-" * 40)

    points_3d = [
        point3d(1, 0, 0),
        point3d(0, 1, 0),
        point3d(0, 0, 1),
        point3d(1, 1, 1),
    ]

    print("Converting Cartesian (x,y,z) to Spherical (r,θ,φ):\n")
    for p in points_3d:
        r, theta, phi = cartesian_to_spherical(p)
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi

        print(f"({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) → "
              f"r={r:.3f}, θ={theta_deg:6.1f}°, φ={phi_deg:6.1f}°")

    # Spherical to Cartesian
    print("\n6. Spherical → Cartesian Conversion (3D)")
    print("-" * 40)

    # Points on unit sphere
    spherical_coords = [
        (1.0, 0.0, np.pi/2),       # +X axis
        (1.0, np.pi/2, np.pi/2),   # +Y axis
        (1.0, 0.0, 0.0),           # +Z axis
        (1.0, np.pi/4, np.pi/4),   # Diagonal
    ]

    print("Converting Spherical (r,θ,φ) to Cartesian (x,y,z):\n")
    for r, theta, phi in spherical_coords:
        p = spherical_to_cartesian(r, theta, phi)
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi

        print(f"r={r:.1f}, θ={theta_deg:6.1f}°, φ={phi_deg:6.1f}° → "
              f"({p.x:6.3f}, {p.y:6.3f}, {p.z:6.3f})")

    # Sphere surface sampling
    print("\n7. Sampling Points on Sphere Surface")
    print("-" * 40)

    radius = 5.0
    num_points = 8

    print(f"Generating {num_points} points on sphere (r={radius}):\n")

    sphere_points = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points  # Azimuthal angle
        phi = np.pi / 3  # Fixed polar angle (60° from north pole)

        p = spherical_to_cartesian(radius, theta, phi)
        sphere_points.append(p)

        print(f"Point {i+1}: ({p.x:6.3f}, {p.y:6.3f}, {p.z:6.3f})")

    # Practical application: Radar/Sonar positioning
    print("\n8. Practical Application: Radar Positioning")
    print("-" * 40)

    print("Radar detects objects in polar coordinates:\n")

    detections = [
        (100.0, np.pi/6, "Aircraft"),      # 100m, 30°
        (50.0, np.pi/2, "Bird"),           # 50m, 90°
        (200.0, 3*np.pi/4, "Cloud"),       # 200m, 135°
    ]

    radar_position = point2d(0, 0)

    for r, theta, label in detections:
        # Convert polar to Cartesian
        p = polar_to_cartesian(r, theta)
        theta_deg = theta * 180 / np.pi

        print(f"{label}:")
        print(f"  Polar:     r={r:6.1f}m, θ={theta_deg:6.1f}°")
        print(f"  Cartesian: ({p.x:6.1f}, {p.y:6.1f})m")

        # Calculate actual distance (should match r)
        dist = distance_point_point(radar_position, p)
        print(f"  Distance:  {dist:.1f}m (verification)\n")

    # Practical application: Geographic coordinates
    print("\n9. Application: Points on Earth's Surface")
    print("-" * 40)

    earth_radius = 6371.0  # km

    # Cities (simplified: lat/lon → spherical)
    # θ = longitude, φ = 90° - latitude
    cities = [
        ("Equator (0°N, 0°E)", 0, np.pi/2),
        ("North Pole (90°N)", 0, 0),
        ("45°N, 45°E", np.pi/4, np.pi/4),
    ]

    print(f"Points on Earth (r={earth_radius}km):\n")

    for name, theta, phi in cities:
        p = spherical_to_cartesian(earth_radius, theta, phi)
        print(f"{name}:")
        print(f"  Cartesian: ({p.x:8.1f}, {p.y:8.1f}, {p.z:8.1f})km\n")

    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
