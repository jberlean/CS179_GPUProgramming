#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "interact_kernel.cuh"

// window
int window_width = 512;
int window_height = 512;

// vbo variables
GLuint vbo_pos[2];

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runParticles(int argc, char** argv);

GLvoid initGL();

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Usage: ./particles_interact N\n");
        exit(1);
    }

    setNumBodies(atoi(argv[1]));
    
    runParticles(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple particle simulation for CUDA
////////////////////////////////////////////////////////////////////////////////
void runParticles(int argc, char** argv)
{
    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow(" Interacting Particles ");

    // Init random number generator
    srand(time(NULL));

    // initialize GL
    initGL();

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // create VBO
    createVBOs(vbo_pos);

    // create other device memory (velocities!)
    createDeviceData();

    // start rendering mainloop
    glutMainLoop();
}


////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
GLvoid initGL()
{
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // TODO (maybe) :: depending on your parameters, you may need to change
    // near and far view distances (1, 500), to better see the simulation.
    // If you do this, probably also change translate_z initial value at top.
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 1, 500.0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    runCuda(vbo_pos, 0.001);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo with newPos
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[getPingpong()]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 1.0, 0.0);
    glDrawArrays(GL_POINTS, 0, getNumBodies());
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        deleteVBOs(vbo_pos);
        deleteDeviceData();
        exit(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
