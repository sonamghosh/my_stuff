// Copyright EC327 2018 Sonam Ghosh sonamg@bu.edu
// Copyright EC327 2018 Nicole Chen nchen357@bu.edu
// Copyright EC327 2018 Tommy Lam tlam11@bu.edu


#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>

using std::ostream;
using std::istream;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

// Class for vectors in 2-dimensional space
class vec2d {
  public:
    double x;
    double y;
    // Default Constructor
    vec2d() {
      x = 0;
      y = 0;
    }
    // Implicit Constructor
    vec2d(double _x, double _y) {
      x = _x;
      y = _y;
    }
    // Calculates the Dot Product with itself
    double sq_mag() const { return dot_product(*this) ; }
    // Calculates the magnitude of a vector
    double mag() const { return std::sqrt(sq_mag()); }
    // Calculates the dot product of two vectors
    double dot_product(const vec2d& other) const {
      return x * other.x + y * other.y;
    }
    // Sets two vectors equal to each other
    bool operator==(const vec2d& other) const {
      return (x == other.x && y == other.y);
    }
    // Adds two vectors
    vec2d operator+(const vec2d& other) const {
      return {x + other.x, y + other.y};
    }
    // Subtracts two vectors
    vec2d operator-(const vec2d& other) const {
      return {x - other.x, y - other.y};
    }
    // Scalar Multiplication to a vector
    vec2d operator*(const double &scalar) const {
      return {x*scalar, y*scalar};
    }
    // Adds to the components of the vector
    vec2d operator+=(const vec2d& other) {
      x += other.x;
      y += other.y;
      return *this;
    }
};

class Stone {
  private:
    double radius;
    vec2d pos;
    vec2d vel;
    double mass;
    string name;

  public:
    // Default Constructor
    Stone() {
      radius = 0;
      pos = {0, 0};
      vel = {0, 0};
      mass = 0;
    }
    // Explicit Constructor
    explicit Stone(const vec2d& _pos, const vec2d& _vel, const double& _mass,
                   const double& _radius, const string& _name) {
              name = _name;
              pos = _pos;
              vel = _vel;
              mass = _mass;
              radius = _radius;
    }

  // Calculates the collision time of stones
  double collision_time(const Stone& s) {
    /**
    @brief Finds the collision time between two stones given the following
           initial conditions:
           1. Starting position vector at t=0 , x_1, x_2
           2. Velocity vector, v_1(t) , v_2 (t)
           3. Radii r_1, r_2

           Define the following vectors:
           x_12 = x_1 - x_2
           v_12 = v_1 - v_2

           Then the time can be found from the following quadratic expression
           0 = t^2 (v_12 * v_12) + 2t (v_12 * x_12) + (x_12 * x_12)-(r_1+r_2)^2
           * indicates the dot product between two vectors
           One can solve for t with the quadratic formula substituting for
           a, b and c.
           a = v_12 * v_12
           b = 2(v_12 * x_12)
           c = (x_12 * x_12 ) - (r_1 + r_2)^2

           The quadratic expression will give three scenarios based on the
           value of the determinant b^2 - 4ac
           (1) b^2 - 4ac < 0 , provides two complex roots that are conjugate
           of each other therefore implying the stones never collide
           (2) b^2 - 4ac = 0, provides one real root , t = -b/2a
           (3) b^2 - 4ac > 0 , provdes two real roots t_1, t_2 but you want
           only the earliest one t = min(t_1, t_2)

    @param s, the reference to a stone object with its associated attributes.

    @return time of collision
    */

    vec2d diff_pos = this->pos - s.pos;
    vec2d diff_vel = this->vel - s.vel;

    // Coefficients of Quadratic Equation
    double a = diff_vel.dot_product(diff_vel);
    double b = 2 * diff_pos.dot_product(diff_vel);
    double c = diff_pos.dot_product(diff_pos) - std::pow(this->radius + s.radius, 2);
    // Solve discriminant
    double det = std::pow(b, 2) - 4 * a * c;
    // Collision Scenarios (refer to above)
    if (det < 0) return -1.0f;
    if (det == 0) return (-b / (2 * a));
    // det > 0 condition
    double t1 = (-b + std::sqrt(det)) / (2 * a);
    double t2 = (-b - std::sqrt(det)) / (2 * a);
    // Prevent negative time
    if (t1 < 0 && t2 >= 0) {
      return t2;
    } else if (t2 < 0 && t1 >= 0) {
      return t1;
    } else if (t2 < 0 && t1 < 0 ) {
      return -1.0f;
    }
    // Choose the smaller of the two times
    double t = std::min(t1, t2);
    return t;
  }

  void move(double time) {
    /**
    @brief Updates the position of the stone after collision
           x = x_0 + v*t
           where x is the updated position, x_0 is the current position
           v is the velocity and t is time

    @param time, amount of time that has passed by to update the stone position
    */

    pos += vel * time;
  }


  void collide(Stone *s) {
    /**
    @brief Updates the velocity of the stones after a collision has occured

    @param s, a pointer to an instance of a Stone object
    */

    // Make sure move is called before this to update the position vector'
    vec2d diff_pos_s1 = this->pos - s->pos;
    vec2d diff_vel_s1 = this->vel - s->vel;
    double mass_ratio_s1 = (2 * s->mass) / (this->mass + s->mass);
    double num_s1 = diff_pos_s1.dot_product(diff_vel_s1);
    double denom_s1 = diff_pos_s1.dot_product(diff_pos_s1);
    vec2d v1 = this->vel - (diff_pos_s1 * (mass_ratio_s1 * (num_s1/denom_s1)));

    vec2d diff_pos_s2 = s->pos - this->pos;
    vec2d diff_vel_s2 = s->vel - this->vel;
    double mass_ratio_s2 = (2 * this->mass) / (this->mass + s->mass);
    double num_s2 = diff_vel_s2.dot_product(diff_pos_s2);
    double denom_s2 = diff_pos_s2.dot_product(diff_pos_s2);
    vec2d v2 = s->vel - (diff_pos_s2 * (mass_ratio_s2 * (num_s2/denom_s2)));

    this->vel = v1;
    s->vel = v2;
  }

  // Compares the names of the stones and puts them in order
  bool operator<(const Stone& other) {
    return name < other.name;
  }

  // Calculates the total energy of the Stone system
  double energy() const {
    return 0.5 * vel.dot_product(vel) * mass;
  }

  // Calculates the total momentum of the Stone system
  vec2d momentum() const {
    return vel * mass;
  }

  // Getters
  double get_radius() const{
    return radius;
  }

  double get_mass() const{
    return mass;
  }

  vec2d get_vel() const{
    return vel;
  }

  vec2d get_pos() const{
    return pos;
  }

  string get_name() const{
    return name;
  }

};

class Collision {
  private:
    double time;

  public:
    Stone *one;
    Stone *two;

    // Default Constructor
    Collision() {
      time = 0;
    }
    // Main Constructor
    Collision(double t, Stone* _one, Stone* _two) {
      time = t;
      one = _one;
      two = _two;
    }

    // Allows calling of the name of stones with <<
    friend std::ostream& operator<<(std::ostream&os, const Collision& i) {
      os << i.one->get_name() << " " << i.two->get_name() << "\n";
    }

    // Checks if a collision is valid
    bool valid() {
      vec2d r = this->one->get_pos() - this->two->get_pos();
      vec2d v = this->two->get_vel() - this->two->get_vel();
      if(r.dot_product(v) < 0) {
        return true;
      } else {
        return false;
      }
    }

    // Getters
    double get_time() const{
      return time;
    }
};



Collision find_fastest_collision(vector<Collision> col) {
  /**
  @brief Find the fastest collision given all collision possibilities

  @param col, a vector of collisions made as Collision objects

  @return result, the fastest collision object
  */

  double ref_time = col.at(0).get_time(); // Collision time
  Collision result = col.at(0);  // Collision object
  double col_t = 0;
  for (int i = 0; i < col.size(); i++) {
    col_t = col.at(i).get_time(); // grab all collision times
    if(col_t < ref_time) {
      ref_time = col_t; // update time
      result = col.at(i); // set the collision to the fastest time
    }
  }
  return result;
}



Collision get_next_collision(vector<Stone> *ps) {
  /**
  @brief Loop through all collisions and give the valid ones

  @param ps, a pointer to a vector of Stone objects

  @return fast, the fastest collision object
  */

  vector<Collision> cols;
  for (int i = 0; i < ps->size(); i++) {
    for (int j = i+1; j < ps->size(); j++) {
      double t = ps->at(i).collision_time(ps->at(j));
      if (ps->at(i) < ps->at(j) && t > 0) {
        Collision new_collision(t, &ps->at(i), &ps->at(j));
        cols.push_back(new_collision);
      }
    }
  }
  Collision fast = find_fastest_collision(cols);
  return fast;
}



void show_stones(vector<Stone> const& stones) {
  /**
  @brief Displays all stones and their attributes

  @param stones, a vector of stone objects
  */

  double energy{0};
  vec2d momentum;
  for (auto &s: stones) {
    cout << s.get_name() << " m=" << s.get_mass() << " R=" << s.get_radius()
    << " p=(" << s.get_pos().x << "," << s.get_pos().y << ") v=("
    << s.get_vel().x << "," << s.get_vel().y << ")" << endl;

    momentum = momentum + s.momentum();
    energy += s.energy();
  }
  cout << "energy: " << std::to_string(energy) << endl;
  cout << "momentum: (" << std::to_string(momentum.x) << ","
  << std::to_string(momentum.y) << ")" << endl;
}



int main() {
  cout << "Hello Prof. Carruthers" << endl;
  cout << "Sorry for being so late for this code Professor :(((" << endl;
  cout << "This was class was fun and taught me a lot about" << endl;
  cout << "Memory stuff and optimization which I hadn't considered" << endl;
  cout << "Much before as someone who did a lot of Python coding" << endl;
  cout << "before this, Have a good summer!" << endl;
  double overall_time = 0;
  cout << "Please enter the x/y position, x/y velocity, mass, radius\n";
  cout << "and name of each stone \n";
  cout << "When complete, use EOF/ Ctrl-D to stop entering \n";

  // Initlaizers
  vector<Stone> stones; // store all Stones
  Stone s; // instance of a stone
  // Paramter Inputs
  string _name;
  double _pos_x, _pos_y, _vel_x, _vel_y;
  double _mass, _radius;
  while (cin >> _pos_x >> _pos_y >> _vel_x >> _vel_y >> _mass >> _radius >> _name) {
    string name = _name;
    double mass = _mass;
    double radius = _radius;
    vec2d pos(_pos_x, _pos_y);
    vec2d vel(_vel_x, _vel_y);
    Stone s(pos, vel, mass, radius, name);
    stones.push_back(s);
  }
  std::sort(stones.begin(), stones.end());

  cout << "\n Here are the initial stones: \n";
  show_stones(stones);

  cout << "\n Here are the collision events \n";

  // Output all the collisions of the system
  while (true) {
    try {
    Collision c = get_next_collision(&stones);

    double new_time = c.get_time();
    for (auto &s: stones)
       s.move(new_time);
    overall_time += new_time;
    cout << "\n time of event:  " << overall_time << endl;
    cout << "colliding " << c;
    c.one->collide(c.two);
    show_stones(stones);
  } catch (std::out_of_range) {}
}

}
