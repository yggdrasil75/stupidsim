def icosphere_faces_vertices(n):
    # Initial icosahedron has 20 faces and 12 vertices
    faces = 20
    vertices = 12
    
    # Each subdivision increases faces and vertices
    for i in range(n):
        faces *= 4
        vertices = vertices + (faces // 2) - vertices + 2
    
    return vertices, faces

def main():
    n = int(input("Enter the subdivision level (n): "))
    vertices, faces = icosphere_faces_vertices(n)
    print(f"At subdivision level {n}, the icosphere has:")
    print(f"- {vertices} vertices")
    print(f"- {faces} faces")

if __name__ == "__main__":
    main()
