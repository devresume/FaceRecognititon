#include <QtWidgets>
#include <QMenu>
#include "imageitem.h"
#include "dialog.h"

imageItem::imageItem( QWidget *parent) : QLabel(parent)
{
}

imageItem::~imageItem()
{

}

void imageItem::mousePressEvent ( QMouseEvent * event )
{

//    imgSel->setStyle("QLabel { background: red; } ");
//    imgSel = this;
//    setStyleSheet("QLabel { background: blue; } ");
//    emit(clicked());

}

void imageItem::setStyle(QString style)
{
    setStyleSheet(style);
}

//void imageItem::contextMenuEvent(QContextMenuEvent *event)
//{
//    if(!imgSel)
//    {
//        imgSel->setStyle("QLabel { background: blue; } ");
//    }
//    setStyle("QLabel { background: red; } ");

////    QMenu* menu = new QMenu(this);
////    QAction* menuAction = new QAction(QString("Remove number: %1").arg(pos), this);
////    menu->addAction(menuAction);
////    QAction* act = menu->exec(event->globalPos());
////    if(act)
////    {
////        ((Dialog*)this->parent())->imgSnapList.removeAt(pos);
////        ((Dialog*)this->parent())->reloadImages();
////    }
//}
