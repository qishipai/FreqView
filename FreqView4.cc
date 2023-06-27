#include <cstdio>
#include <cmath>
#include <cstring>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#ifdef __ANDROID__
#include <GLES3/gl31.h>
#else
#include <GL/glew.h>
#endif





GLuint gl_CreateShader(const char *vsrc, const char *fsrc)
{
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint frgmnt_shader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint shader_program = glCreateProgram();
    static char errstr[800];
    GLint status, lstatus;

    glShaderSource(vertex_shader, 1, &vsrc, NULL);
    glShaderSource(frgmnt_shader, 1, &fsrc, NULL);

    glCompileShader(vertex_shader);
    glAttachShader(shader_program, vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);

    if (!status)
    {
        glGetShaderInfoLog(vertex_shader, 800, NULL, errstr);
        printf("Vert-Shader Compile FAILED >>>\n%s\n", errstr);
        return 0;
    }

    glCompileShader(frgmnt_shader);
    glAttachShader(shader_program, frgmnt_shader);
    glGetShaderiv(frgmnt_shader, GL_COMPILE_STATUS, &status);

    if (!status)
    {
        glGetShaderInfoLog(frgmnt_shader, 800, NULL, errstr);
        printf("Frag-Shader Compile FAILED >>>\n%s\n", errstr);
        return 0;
    }

    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &lstatus);

    if (!lstatus)
    {
        glGetProgramInfoLog(shader_program, 800, NULL, errstr);
        printf("Shader Program Link ERROR >>>\n%s\n", errstr);
        return 0;
    }

    return shader_program;
}



void gl_LoadTexture2D(SDL_Surface *sur)
{
    SDL_Surface *rgba = SDL_ConvertSurfaceFormat(sur, SDL_PIXELFORMAT_RGBA32, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba->w, rgba->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba->pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    SDL_FreeSurface(rgba);
}




void mat4Copy(float m[], const float m1[])
{
    m[ 0] = m1[ 0], m[ 1] = m1[ 1], m[ 2] = m1[ 2], m[ 3] = m1[ 3];
    m[ 4] = m1[ 4], m[ 5] = m1[ 5], m[ 6] = m1[ 6], m[ 7] = m1[ 7];
    m[ 8] = m1[ 8], m[ 9] = m1[ 9], m[10] = m1[10], m[11] = m1[11];
    m[12] = m1[12], m[13] = m1[13], m[14] = m1[14], m[15] = m1[15];
}

void mat4Mul(float m[], const float m1[], const float m2[])
{
    m[ 0] =  m1[ 0] * m2[ 0] + m1[ 1] * m2[ 4] + m1[ 2] * m2[ 8] + m1[ 3] * m2[12];
    m[ 1] =  m1[ 0] * m2[ 1] + m1[ 1] * m2[ 5] + m1[ 2] * m2[ 9] + m1[ 3] * m2[13];
    m[ 2] =  m1[ 0] * m2[ 2] + m1[ 1] * m2[ 6] + m1[ 2] * m2[10] + m1[ 3] * m2[14];
    m[ 3] =  m1[ 0] * m2[ 3] + m1[ 1] * m2[ 7] + m1[ 2] * m2[11] + m1[ 3] * m2[15];
    m[ 4] =  m1[ 4] * m2[ 0] + m1[ 5] * m2[ 4] + m1[ 6] * m2[ 8] + m1[ 7] * m2[12];
    m[ 5] =  m1[ 4] * m2[ 1] + m1[ 5] * m2[ 5] + m1[ 6] * m2[ 9] + m1[ 7] * m2[13];
    m[ 6] =  m1[ 4] * m2[ 2] + m1[ 5] * m2[ 6] + m1[ 6] * m2[10] + m1[ 7] * m2[14];
    m[ 7] =  m1[ 4] * m2[ 3] + m1[ 5] * m2[ 7] + m1[ 6] * m2[11] + m1[ 7] * m2[15];
    m[ 8] =  m1[ 8] * m2[ 0] + m1[ 9] * m2[ 4] + m1[10] * m2[ 8] + m1[11] * m2[12];
    m[ 9] =  m1[ 8] * m2[ 1] + m1[ 9] * m2[ 5] + m1[10] * m2[ 9] + m1[11] * m2[13];
    m[10] =  m1[ 8] * m2[ 2] + m1[ 9] * m2[ 6] + m1[10] * m2[10] + m1[11] * m2[14];
    m[11] =  m1[ 8] * m2[ 3] + m1[ 9] * m2[ 7] + m1[10] * m2[11] + m1[11] * m2[15];
    m[12] =  m1[12] * m2[ 0] + m1[13] * m2[ 4] + m1[14] * m2[ 8] + m1[15] * m2[12];
    m[13] =  m1[12] * m2[ 1] + m1[13] * m2[ 5] + m1[14] * m2[ 9] + m1[15] * m2[13];
    m[14] =  m1[12] * m2[ 2] + m1[13] * m2[ 6] + m1[14] * m2[10] + m1[15] * m2[14];
    m[15] =  m1[12] * m2[ 3] + m1[13] * m2[ 7] + m1[14] * m2[11] + m1[15] * m2[15];
}

void mat4I(float mat[])
{
    mat[ 0] = 1, mat[ 1] = mat[ 2] = mat[ 3] = mat[ 4] = 0;
    mat[ 5] = 1, mat[ 6] = mat[ 7] = mat[ 8] = mat[ 9] = 0;
    mat[10] = 1, mat[11] = mat[12] = mat[13] = mat[14] = 0;
    mat[15] = 1;
}

void mat4RotaX(float mat[], float a)
{
    float c = cosf(a), s = sinf(a), tmp[16];

    float m[16] = {
      1.0, 0.0, 0.0, 0.0,
      0.0,   c,  -s, 0.0,
      0.0,   s,   c, 0.0,
      0.0, 0.0, 0.0, 1.0,
    };

    mat4Mul(tmp, m, mat), mat4Copy(mat, tmp);
}

void mat4RotaY(float mat[], float a)
{
    float c = cosf(a), s = sinf(a), tmp[16];

    float m[16] = {
        c, 0.0,  -s, 0.0,
      0.0, 1.0, 0.0, 0.0,
        s, 0.0,   c, 0.0,
      0.0, 0.0, 0.0, 1.0,
    };

    mat4Mul(tmp, m, mat), mat4Copy(mat, tmp);
}

void mat4RotaZ(float mat[], float a)
{
    float c = cosf(a), s = sinf(a), tmp[16];

    float m[16] = {
        c,  -s, 0.0, 0.0,
        s,   c, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
    };

    mat4Mul(tmp, m, mat), mat4Copy(mat, tmp);
}








struct MD_OBJECT {

  virtual ~MD_OBJECT() = default;

  virtual void DrawObj() = 0;

  virtual void setMatrix(float mat[16]) = 0;

  virtual void setProj(float mat[16]) = 0;

  virtual void setView(float mat[16]) = 0;
};




/* ========== [ 背景坐标网格 ] ========== */

#define LINE "\n"
#ifdef __ANDROID__
const char *bgmesh_vshader = "#version 320 es"
#else
const char *bgmesh_vshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"layout(location = 0) in vec3 vertex;"
LINE"layout(location = 1) in vec3 vcolor;"
LINE"uniform mat4 mat;"
LINE"out vec4 lineColor;"

LINE"void main() {"
LINE"  lineColor = vec4(vcolor, 1);"
LINE"  gl_Position = mat * vec4(vertex, 1);"
LINE"}";

#ifdef __ANDROID__
const char *bgmesh_fshader = "#version 320 es"
#else
const char *bgmesh_fshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"in  vec4 lineColor;"
LINE"out vec4 fragColor;"

LINE"void main() { fragColor = lineColor; }";


struct MD_COORD : public MD_OBJECT {
  GLuint shader, VAO, VBO, loc_m, lines;
  float proj[16], view[16];

  MD_COORD(int Ndiv, float range)
  {
      shader = gl_CreateShader(bgmesh_vshader, bgmesh_fshader);

      if (!shader) { return; }

      loc_m = glGetUniformLocation(shader, "mat");

      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);

      lines = (Ndiv * 2 + 1) * 2;
      GLfloat *vtx = new GLfloat [lines * 12];

      for (int p = -Ndiv, pv = 0; p <= Ndiv; ++p, ++pv)
      {
          float w = (float)range * p / Ndiv;

          vtx[pv * 12 + 0] = vtx[pv * 12 + 6] = w;
          vtx[pv * 12 + 1] = range, vtx[pv * 12 + 7] = -range;
          vtx[pv * 12 + 2] = vtx[pv * 12 + 8] = 0;

          if (p == 0)
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 0.5;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 1.0;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.5;
          }
          else if (p % 10 == 0)
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 0.7;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 0.7;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.3;
          }
          else
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 0.4;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 0.4;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.4;
          }

          ++pv;

          vtx[pv * 12 + 0] = range, vtx[pv * 12 + 6] = -range;
          vtx[pv * 12 + 1] = vtx[pv * 12 + 7] = w;
          vtx[pv * 12 + 2] = vtx[pv * 12 + 8] = 0;

          if (p == 0)
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 1.0;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 0.5;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.5;
          }
          else if (p % 10 == 0)
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 0.7;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 0.7;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.3;
          }
          else
          {
              vtx[pv * 12 + 3] = vtx[pv * 12 +  9] = 0.4;
              vtx[pv * 12 + 4] = vtx[pv * 12 + 10] = 0.4;
              vtx[pv * 12 + 5] = vtx[pv * 12 + 11] = 0.4;
          }
      }

      glBindVertexArray(VAO);
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      glBindBuffer(GL_ARRAY_BUFFER, VBO);

      glBufferData(GL_ARRAY_BUFFER,
        sizeof(GLfloat) * lines * 12, vtx, GL_STATIC_DRAW);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 6, (void *)(sizeof(GLfloat) * 0));

      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 6, (void *)(sizeof(GLfloat) * 3));

      delete[] vtx;
      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  ~MD_COORD()
  {
      glDeleteBuffers(1, &VBO);
      glDeleteVertexArrays(1, &VAO);
  }

  void DrawObj()
  {
      glUseProgram(shader);
      glBindVertexArray(VAO);
      glDrawArrays(GL_LINES, 0, lines * 2);
      glBindVertexArray(0);
  }

  void setMatrix(float mat[16])
  {
      float m[16];
      mat4Mul(m, proj, view);
      glProgramUniformMatrix4fv(shader, loc_m, 1, GL_TRUE, m);
  }

  void setProj(float mat[16]) { mat4Copy(proj, mat); }

  void setView(float mat[16]) { mat4Copy(view, mat); }
};




/* ========== [ 简单纹理标牌 ] ========== */

#ifdef __ANDROID__
const char *mlabel_vshader = "#version 320 es"
#else
const char *mlabel_vshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"layout(location = 0) in vec3 vertex;"
LINE"layout(location = 1) in vec2 tcoord;"
LINE"uniform mat4 mat;"
LINE"out vec2 txcoord;"

LINE"void main() {"
LINE"  txcoord = tcoord;"
LINE"  gl_Position = mat * vec4(vertex, 1);"
LINE"}";

#ifdef __ANDROID__
const char *mlabel_fshader = "#version 320 es"
#else
const char *mlabel_fshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"uniform sampler2D lb_tx;"
LINE"in  vec2 txcoord;"
LINE"out vec4 fragColor;"

LINE"void main() {"
LINE"  fragColor = texture(lb_tx, txcoord);"
LINE"}";


struct MD_LABEL : public MD_OBJECT {
  GLuint shader, VAO, VBO, texture, loc_m;
  float proj[16], view[16];

  MD_LABEL()
  {
      shader = gl_CreateShader(mlabel_vshader, mlabel_fshader);

      if (!shader) { return; }

      loc_m = glGetUniformLocation(shader, "mat");

      glGenVertexArrays(1, &VAO);
      glGenTextures(1, &texture);
      glGenBuffers(1, &VBO);

      glBindVertexArray(VAO);
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glBindVertexArray(0);
  }

  void loadSurface(SDL_Surface *sur, float px_scale)
  {
      float sw = sur->w*px_scale/2, sh = sur->h*px_scale/2;

      glBindTexture(GL_TEXTURE_2D, texture);
      gl_LoadTexture2D(sur);
      glBindTexture(GL_TEXTURE_2D, 0);

      GLfloat quad_vtx[30] = {
        -sw, -sh, 0.0, 0.0, 1.0,  sw, -sh, 0.0, 1.0, 1.0,
         sw,  sh, 0.0, 1.0, 0.0, -sw, -sh, 0.0, 0.0, 1.0,
         sw,  sh, 0.0, 1.0, 0.0, -sw,  sh, 0.0, 0.0, 0.0,
      };

      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);

      glBufferData(GL_ARRAY_BUFFER,
        sizeof(GLfloat) * 30, quad_vtx, GL_STREAM_DRAW);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 5, (void *)(sizeof(GLfloat) * 0));

      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 5, (void *)(sizeof(GLfloat) * 3));

      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  ~MD_LABEL()
  {
      glDeleteBuffers(1, &VBO);
      glDeleteTextures(1, &texture);
      glDeleteVertexArrays(1, &VAO);
  }

  void DrawObj()
  {
      glUseProgram(shader);
      glBindVertexArray(VAO);
      glBindTexture(GL_TEXTURE_2D, texture);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
  }

  void setMatrix(float mat[16])
  {
      float m1[16], m2[16];
      mat4Mul(m1, proj, view), mat4Mul(m2, m1, mat);
      glProgramUniformMatrix4fv(shader, loc_m, 1, GL_TRUE, m2);
  }

  void setProj(float mat[16]) { mat4Copy(proj, mat); }

  void setView(float mat[16]) { mat4Copy(view, mat); }
};




/* ========== [ 简单光照方块 ] ========== */

#ifdef __ANDROID__
const char *cube0_vshader = "#version 320 es"
#else
const char *cube0_vshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"layout(location = 0) in vec3 vertex;"
LINE"layout(location = 1) in vec3 vtnorm;"
LINE"uniform vec3 light;"
LINE"uniform vec4 color;"
LINE"uniform mat4 proj;"
LINE"uniform mat4 view;"
LINE"uniform mat4 model;"
LINE"out vec4 surfColor;"

LINE"void main() {"
LINE"  vec3 vt_Position = (view * model * vec4(vertex, 1)).xyz;"
LINE"  vec3 n = normalize((view * model * vec4(vtnorm, 0)).xyz);"
LINE"  vec3 s = normalize((view * vec4(light, 1)).xyz - vt_Position);"

LINE"  float li = 0.62 "
  "+ max(dot(s,n),0) * 0.4"
  "+ pow(max(dot(normalize(-vt_Position),reflect(-s,n)),0),6) * 0.3;"

LINE"  surfColor = vec4(color.rgb * li, color.a);"
LINE"  gl_Position = proj * vec4(vt_Position, 1);"
LINE"}";

#ifdef __ANDROID__
const char *cube0_fshader = "#version 320 es"
#else
const char *cube0_fshader = "#version 330"
#endif
LINE"precision mediump float;"
LINE"in  vec4 surfColor;"
LINE"out vec4 fragColor;"

LINE"void main() { fragColor = surfColor; }";


struct MD_CUBE_0 : public MD_OBJECT {
  GLuint shader, VAO, VBO, EBO, ecnt, loc_proj;
  GLuint loc_view, loc_model, loc_light, loc_color;

  MD_CUBE_0()
  {
      shader = gl_CreateShader(cube0_vshader, cube0_fshader);

      if (!shader) { return; }

      loc_proj = glGetUniformLocation(shader, "proj");
      loc_view = glGetUniformLocation(shader, "view");
      loc_model = glGetUniformLocation(shader, "model");
      loc_light = glGetUniformLocation(shader, "light");
      loc_color = glGetUniformLocation(shader, "color");

      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);
      glGenBuffers(1, &EBO);

      glBindVertexArray(VAO);
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      GLuint cube_index[36] = {
         0,  1,  2,  0,  2,  3,  4,  5,  6,  4,  6,  7,
         8,  9, 10,  8, 10, 11, 12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
      };

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

      glBufferData(GL_ELEMENT_ARRAY_BUFFER,
        sizeof(GLuint) * 36, cube_index, GL_STATIC_DRAW);

      glBindVertexArray(0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  ~MD_CUBE_0()
  {
      glDeleteBuffers(1, &VBO);
      glDeleteBuffers(1, &EBO);
      glDeleteVertexArrays(1, &VAO);
  }

  void setSize(float x, float y, float z)
  {
      GLfloat cube_vertex[] = {
        0, 0, 0,  0, -1,  0, x, 0, 0,  0, -1,  0,
        x, 0, z,  0, -1,  0, 0, 0, z,  0, -1,  0,

        x, 0, 0,  1,  0,  0, x, y, 0,  1,  0,  0,
        x, y, z,  1,  0,  0, x, 0, z,  1,  0,  0,

        x, y, 0,  0,  1,  0, 0, y, 0,  0,  1,  0,
        0, y, z,  0,  1,  0, x, y, z,  0,  1,  0,

        0, y, 0, -1,  0,  0, 0, 0, 0, -1,  0,  0,
        0, 0, z, -1,  0,  0, 0, y, z, -1,  0,  0,

        0, 0, z,  0,  0,  1, x, 0, z,  0,  0,  1,
        x, y, z,  0,  0,  1, 0, y, z,  0,  0,  1,

        x, 0, 0,  0,  0, -1, 0, 0, 0,  0,  0, -1,
        0, y, 0,  0,  0, -1, x, y, 0,  0,  0, -1,
      };

      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);

      glBufferData(GL_ARRAY_BUFFER,
        sizeof(cube_vertex), cube_vertex, GL_STREAM_DRAW);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 6, (void *)(sizeof(GLfloat) * 0));

      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
        sizeof(GLfloat) * 6, (void *)(sizeof(GLfloat) * 3));

      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  void setColor(float r, float g, float b, float a)
  {
      glProgramUniform4f(shader, loc_color, r, g, b, a);
  }

  void setLignt(float lx, float ly, float lz)
  {
      glProgramUniform3f(shader, loc_light, lx, ly, lz);
  }

  void DrawObj()
  {
      glUseProgram(shader);
      glBindVertexArray(VAO);
      glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, NULL);
      glBindVertexArray(0);
  }

  void setMatrix(float mat[16])
  {
      glProgramUniformMatrix4fv(shader, loc_model, 1, GL_TRUE, mat);
  }

  void setProj(float mat[16])
  {
      glProgramUniformMatrix4fv(shader, loc_proj, 1, GL_TRUE, mat);
  }

  void setView(float mat[16])
  {
      glProgramUniformMatrix4fv(shader, loc_view, 1, GL_TRUE, mat);
  }
};







void gl_ProjMat(float mat[16], float d, float f, float w)
{
    mat[ 0] = d*w;
    mat[ 1] = 0;
    mat[ 2] = 0;
    mat[ 3] = 0;

    mat[ 4] = 0;
    mat[ 5] = 0;
    mat[ 6] = d;
    mat[ 7] = 0;

    mat[ 8] = 0;
    mat[ 9] = (f+d) / (f-d);
    mat[10] = 0;
    mat[11] = 2*f*d / (d-f);

    mat[12] = 0;
    mat[13] = 1;
    mat[14] = 0;
    mat[15] = 0;
}

void gl_ViewMat(float mat[16], float a, float b, float x, float y, float z)
{
    mat4I(mat);
    mat[3] += x;
    mat4RotaZ(mat, a);
    mat4RotaX(mat, b);
    mat[7] += y, mat[11] += z;
}




#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

using std::sin, std::cos, std::sqrt;


template<typename Tp> struct Complex {
    Tp re, im;

    Complex(Tp _r, Tp _i): re(_r), im(_i) { }

    Complex() = default;

    Complex operator+(const Complex &c) const
    {
        return Complex(re + c.re, im + c.im);
    }

    Complex operator-(const Complex &c) const
    {
        return Complex(re - c.re, im - c.im);
    }

    Complex operator*(const Complex &c) const
    {
        return Complex(re * c.re - im * c.im,
                       re * c.im + im * c.re);
    }
};


template<typename Tp> struct FastDFTctx {
    int fftLen, (*swpIdx)[2];
    Complex<Tp>  *vecUnit[2];
    Tp  adjC;

    void createCtx(int samplN)
    {
        int idx = 0, s = 0, N = fftLen = samplN;
        vecUnit[0] = new Complex<Tp> [fftLen];
        vecUnit[1] = *vecUnit + (fftLen >> 1);
        swpIdx = new int [fftLen >> 1][2];
        adjC = (Tp)1.0 / fftLen;

        for (int i = 1; i < fftLen - 1; ++i)
        {
            s ^= N - N / ((i ^ (i - 1)) + 1);

            if (i < s)
            {
                swpIdx[idx][0] = i;
                swpIdx[idx][1] = s, ++idx;
            }
        }

        swpIdx[idx][0] = 0, s = fftLen >> 1;
        Complex<Tp> *vec_arr = vecUnit[0];
        Tp ang = (Tp)M_PI * 2 / fftLen;

        for (int i = 0; i < s; ++i)
        {
            vec_arr[i | s].re = cos(ang * i);
            vec_arr[i | s].im = sin(ang * i);
            vec_arr[i].re =  vec_arr[i | s].re;
            vec_arr[i].im = -vec_arr[i | s].im;
        }
    }

    void destroyCtx()
    {
        delete[] swpIdx, delete[] vecUnit[0];
    }

    void execTrans(Complex<Tp> *arr, bool m) const
    {
        for (int (*sp)[2] = swpIdx; **sp; ++sp)
        {
            Complex<Tp> t = arr[(*sp)[0]];
            arr[(*sp)[0]] = arr[(*sp)[1]];
            arr[(*sp)[1]] = t;
        }

        for (int sc = 2; sc <= fftLen; sc <<= 1)
        {
            int hs = sc >> 1, d = fftLen / sc;

            for (int t = 0; t < fftLen; t += sc)
                for (int k = 0; k < hs; ++k)
                {
                    int u = k | t, v = u | hs;
                    Complex<Tp> x = arr[u], y;
                    y = arr[v] * vecUnit[m][k * d];
                    arr[u] = x + y, arr[v] = x - y;
                }
        }

        for (int i = 0; m && i < fftLen; ++i)
            arr[i].re *= adjC, arr[i].im *= adjC;
    }
};


template<typename Tp> struct Filter {
    Tp sig_x[4];

    Filter(): sig_x{0, 0, 0, 0} { }

    void sendSig(Tp sig_in)
    {
        sig_x[0] = sig_x[1], sig_x[1] = sig_x[2];
        sig_x[2] = sig_x[3], sig_x[3] = sig_in;
    }

    Tp getSig() const
    {
        return (sig_x[0] + sig_x[1] + sig_x[2] + sig_x[3]) / 4;
    }
};


#define FFT_LEN 2048
#define SCR_W   1920
#define SCR_H   768
#define CC_LEN  32

struct PIX_t {
    unsigned char r, g, b;
};

float ccache[1024][CC_LEN];

static char SCR[SCR_W*SCR_H][4];


static PIX_t genPIX(int key)
{
    PIX_t p = {0, 0, 0};

    switch((key %= 1530) / 255)
    {
    case 0: p.r = 255, p.g = key;        break;
    case 1: p.g = 255, p.r = 510 - key;  break;
    case 2: p.g = 255, p.b = key - 510;  break;
    case 3: p.b = 255, p.g = 1020 - key; break;
    case 4: p.b = 255, p.r = key - 1020; break;
    case 5: p.r = 255, p.b = 1530 - key; break;
    }

    return p;
}


static inline float winFunc(int i)
{
    return 1.0f - cosf(M_PI * i * 2 / (FFT_LEN - 1));
}

/*

static inline float getFreqVal(Complex<float> c)
{
    return powf(c.re * c.re + c.im * c.im, 0.2) * 0.1;
}

*/

static inline float getFreqVal(Complex<float> c)
{
    float u = 0.13, a = c.re * c.re + c.im * c.im;
    return (a*u < 2.71828? u*u*a/2.71828 : u*log(u*a));
}


static char strbuf[700], filename[500];

int main(int ac, char **av)
{
    if (ac != 2)
        return 1;

    strcpy(filename, av[1]);

    int p = strlen(filename) - 1;

    while (p > 0 && filename[p] != '.')
        --p;

    if (p > 0)
        filename[p--] = 0;

    p = strlen(filename) - 1;

    while (p > 0 && filename[p] != '/' && filename[p] != '\\')
        --p;

    strcpy(filename, filename + p + (filename[p] == '/' || filename[p] == '\\'));

    sprintf(strbuf, "ffmpeg -y -hide_banner -f rawvideo -pix_fmt rgba -r 60 -s 1920x768 -i - -i \"%s\" -vf vflip -pix_fmt yuv420p FreqView.mp4", av[1]);

    FILE *vid = popen(strbuf, "wb");

    sprintf(strbuf, "ffmpeg -v error -i \"%s\" -ac 1 -ar 48000 -f f32le -", av[1]);

    FILE *aud = popen(strbuf, "rb");

    SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO);
    TTF_Init();

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE,   8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,  8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_Window *win = SDL_CreateWindow("FreqView4", 200, 100, SCR_W, SCR_H, SDL_WINDOW_OPENGL);

    TTF_Font *ttf = TTF_OpenFont("font.ttf", 80);

    if (!ttf)
    {
        puts("font.ttf NOT FOUND");
        getchar();
        SDL_DestroyWindow(win);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);


#ifndef __ANDROID__
    glewExperimental = true;

    if (glewInit() != GLEW_OK)
        return -1;
#endif


    glEnable(GL_BLEND);
    //glEnable(GL_MULTISAMPLE);
    //glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glViewport(0, 0, SCR_W, SCR_H);

    glLineWidth(1.0f);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClearDepthf(1.0f);


    MD_COORD  bgmesh(40, 4);
    MD_LABEL  mlabel;
    MD_CUBE_0 ntcube;

    ntcube.setLignt(20, -10, 20);

    float proj[16], view[16];
    gl_ProjMat(proj, 3, 80, (float)SCR_H/SCR_W);

    bgmesh.setProj(proj);
    mlabel.setProj(proj);
    ntcube.setProj(proj);

    float ax = 0.2, ay = 0.5, cx = 0, cy = 0, cz = 10, t = 0;
    bool run = true, cap = false;

    gl_ViewMat(view, ax, ay, cx, cz, cy);
    bgmesh.setView(view);
    mlabel.setView(view);
    ntcube.setView(view);

    Complex<float> *arr = new Complex<float> [FFT_LEN];
    Filter <float> *flt = new Filter <float> [FFT_LEN / 2];
    float  *in_buf  = new float  [FFT_LEN] {0};

    FastDFTctx<float>  FFT;
    FFT.createCtx(FFT_LEN);

    int ret;

    do
    {
        if (cap)
        {
            for (int i = 0; i < FFT_LEN - 800; ++i)
                in_buf[i] = in_buf[i + 800];

            ret = fread(in_buf + FFT_LEN - 800, 4, 800, aud);

            for (int i = 0; i < FFT_LEN; ++i)
            {
                arr[i].re = in_buf[i] * winFunc(i);
                arr[i].im = 0;
            }

            FFT.execTrans(arr, 0);

            for (int i = 0; i < FFT_LEN / 2; ++i)
                flt[i].sendSig(getFreqVal(arr[i]));


            ay += cos(t/10) * 2e-4;
            ax += sin(t/7) * 2e-4;

            gl_ViewMat(view, ax, ay, cx, cz, cy);
            bgmesh.setView(view);
            mlabel.setView(view);
            ntcube.setView(view);
        }

        float m[16];

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        mat4I(m);
        m[3] += -1.5;
        m[7] += -2.5;
        m[11] += 0.01;
        bgmesh.setMatrix(m);
        mlabel.setMatrix(m);
        bgmesh.DrawObj();

        gl_ViewMat(view, ax, ay, cx, cz, cy);
        bgmesh.setView(view);
        mlabel.setView(view);
        ntcube.setView(view);

        int _x = 0, c = 1, cn = 0;
        float sy = 0;

        for (int i = 4; i <= FFT_LEN / 2; ++i)
        {
            if (cap)
                sy += flt[i].getSig();

            int x = (log10f(i) - 0.5f) * 325;

            if (x != _x || (++c, false))
            {
                float y = sy / c;

                if (y > 3) y = 3;
    
                if (y < 0) y = 0;

                mat4I(m);
                m[3] = (float)_x / 100 - 4.1;

                if (cap)
                {
                    for (int j = 1; j < CC_LEN; ++j)
                        ccache[cn][j - 1] = ccache[cn][j];

                    ccache[cn][CC_LEN - 1] = y;
                }

                for (int j = 0; j < CC_LEN; ++j)
                {
                    PIX_t pix = genPIX(ccache[cn][j] * 900);
                    ntcube.setColor(pix.r/255.0, pix.g/255.0, pix.b/255.0, 0.8);
                    ntcube.setSize((x - _x) / 100.0, 0.1, ccache[cn][j]);

                    m[7] = 3.7 - (float)j * 0.15;
                    ntcube.setMatrix(m), ntcube.DrawObj();
                }

                _x = x, sy = 0, c = 1, ++cn;
            }
        }

        char str[50];
        int s = t;
        sprintf(str, "FreqView4\n%02d:%02d:%02d", s/3600, s/60%60, s%60);
        SDL_Surface *sur = TTF_RenderUTF8_Blended_Wrapped(ttf, str, {240, 225, 130, 200}, 0);
        mat4I(m);
        mat4RotaX(m, 1.57);
        m[3] += 2;
        m[7] += -3;
        m[11] += 0.8;
        mlabel.setMatrix(m);
        mlabel.loadSurface(sur, 0.005);
        mlabel.DrawObj();
        SDL_FreeSurface(sur);


        PIX_t color = genPIX(1000 + 1000 * sin(t/3.6));
        sur = TTF_RenderUTF8_Blended(ttf, filename, {color.r, color.g, color.b, 200});
        mat4I(m);
        mat4RotaX(m, 1.57);
        mat4RotaY(m, 0.05 * cos(t));
        m[3] += -2;
        m[7] += -3;
        m[11] += 0.8 + 0.05 * sin(t / 2);
        mlabel.setMatrix(m);
        mlabel.loadSurface(sur, 0.005);
        mlabel.DrawObj();
        SDL_FreeSurface(sur);


        mat4I(m);
        m[3] -= 0.25;
        m[7] -= 0.25;
        m[11] -= 0.25;
        mat4RotaX(m, t / 2);
        mat4RotaY(m, t / 3);
        mat4RotaZ(m, t / 2);
        m[3] += 0.2;
        m[7] += -3;
        m[11] += 0.8;
        ntcube.setSize(0.5, 0.5, 0.5);
        ntcube.setColor(0.8, 0.8, 0.8, 1);
        ntcube.setMatrix(m);
        ntcube.DrawObj();


        if (cap)
        {
            glReadPixels(0, 0, SCR_W, SCR_H, GL_RGBA, GL_UNSIGNED_BYTE, SCR);
            fwrite(SCR, 4 * SCR_W * SCR_H, 1, vid);
            t += 1.0 / 60;
        }

        SDL_GL_SwapWindow(win);

        SDL_Event e;

        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                run = false;

            if (e.type == SDL_KEYDOWN)
            {
              switch (e.key.keysym.sym)
              {
                case SDLK_RIGHT:
                  ax -= M_PI / 200;
                  goto change_view;
                case SDLK_LEFT:
                  ax += M_PI / 200;
                  goto change_view;
                case SDLK_UP:
                  ay += M_PI / 200;
                  goto change_view;
                case SDLK_DOWN:
                  ay -= M_PI / 200;
                  goto change_view;
                case SDLK_w:
                  cz -= 0.1;
                  goto change_view;
                case SDLK_s:
                  cz += 0.1;
                  goto change_view;
                case SDLK_a:
                  cx -= 0.1;
                  goto change_view;
                case SDLK_d:
                  cx += 0.1;
                  goto change_view;
                case SDLK_q:
                  cy -= 0.1;
                  goto change_view;
                case SDLK_e:
                  cy += 0.1;
                  goto change_view;
                case SDLK_SPACE:
                  cap = !cap;
                  ret = 800;
                  break;

                change_view:
                  gl_ViewMat(view, ax, ay, cx, cz, cy);
                  bgmesh.setView(view);
                  mlabel.setView(view);
                  ntcube.setView(view);
                  break;
              }
            }
        }

    } while ((!cap || ret == 800) && run);

    TTF_CloseFont(ttf);
    delete[] arr, delete[] flt, delete[] in_buf;
    FFT.destroyCtx(), pclose(vid), pclose(aud);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
