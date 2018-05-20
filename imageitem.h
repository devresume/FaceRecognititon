#ifndef IMAGEITEM_H
#define IMAGEITEM_H

#include <QWidget>
#include <QLabel>
#include <QList>

class imageItem : public QLabel
{
    Q_OBJECT
public:
    explicit imageItem(QWidget *parent = 0);
    ~imageItem();
    int pos;
    imageItem* imgSel;
    QList<imageItem*> items;
    void setStyle(QString style);
protected:
    //void contextMenuEvent(QContextMenuEvent *event) Q_DECL_OVERRIDE;
    void mousePressEvent ( QMouseEvent * event );
signals:
    void clicked();

public slots:
};

#endif // IMAGEITEM_H
